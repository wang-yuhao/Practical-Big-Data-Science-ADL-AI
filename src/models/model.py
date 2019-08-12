#!/usr/bin/python3
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.dataloader import TestDataset


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False,
                 double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        if double_entity_embedding:
            self.entity_dim = hidden_dim*2
        else:
            self.entity_dim = hidden_dim

        if double_relation_embedding:
            self.relation_dim = hidden_dim*2
        else:
            self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity,
                                             self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation,
                                                           self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            tmp = [[0.5 * self.embedding_range.item()]]
            self.modulus = nn.Parameter(torch.Tensor(tmp))

        # Do not forget to modify this line when you add a new model
        # in the "forward" function
        if model_name not in ['RotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding
                                       or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

    def change_entity_embedding(self, new_embedding):
        self.entity_embedding = new_embedding

    def change_relation_embedding(self, new_embedding):
        self.relation_embedding = new_embedding

    #pylint: disable=arguments-differ
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share 2 elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, neg_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, neg_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, neg_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, neg_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, neg_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'RotatE': self.RotatE,
            # 'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def RotatE(self, head, relation, tail, mode):  # often just named "model"
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    # def pRotatE(self, head, relation, tail, mode):
    #     pi = 3.14159262358979323846

    #     # Make phases of entities and relations uniformly
    #     # distributed in [-pi, pi]

    #     phase_head = head/(self.embedding_range.item()/pi)
    #     phase_relation = relation/(self.embedding_range.item()/pi)
    #     phase_tail = tail/(self.embedding_range.item()/pi)

    #     if mode == 'head-batch':
    #         score = phase_head + (phase_relation - phase_tail)
    #     else:
    #         score = (phase_head + phase_relation) - phase_tail

    #     score = torch.sin(score)
    #     score = torch.abs(score)

    #     score = self.gamma.item() - score.sum(dim = 2) * self.modulus
    #     return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        pos_sample, neg_sample, subsampling_weight, mode = next(train_iterator)
        if args.cuda:
            pos_sample = pos_sample.cuda()
            neg_sample = neg_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((pos_sample, neg_sample), mode=mode)

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation
            # on the sampling weight
            negative_score = (F.softmax(negative_score
                              * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(pos_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        value1 = (subsampling_weight * positive_score).sum()
        value2 = subsampling_weight.sum()
        pos_sample_loss = - value1 / value2
        value1 = (subsampling_weight * negative_score).sum()
        neg_sample_loss = - value1 / value2

        loss = (pos_sample_loss + neg_sample_loss)/2
        regularization_log = {}
        loss.backward()
        optimizer.step()

        log = {
            **regularization_log,
            'pos_sample_loss': pos_sample_loss.item(),
            'neg_sample_loss': neg_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args,
                  save_scores=False):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()

        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'head-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'tail-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []
        all_scores = []
        all_scores_filter_bias = []
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset, missing in zip(test_dataset_list, [0, 2]):
                # missing: 0 = head, 2 = tail
                line = 0
                for pos_sample, neg_sample, filter_bias, mode in test_dataset:
                    if args.cuda:
                        pos_sample = pos_sample.cuda()  # shape = (4,3)
                        # 3 because its a triple
                        neg_sample = neg_sample.cuda()  # shape = (4,14541)
                        filter_bias = filter_bias.cuda()  # shape = (4,14541)
                    batch_size = pos_sample.size(0)

                    # score: for each entity a score (negative value or 0!)
                    # index of score = entity
                    # score.shape = (4, 14541) (14541 = num different entities)
                    score_without_bias = model((pos_sample, neg_sample), mode)
                    score = score_without_bias + filter_bias

                    if save_scores:
                        for i in range(batch_size):
                            all_scores.append((line, missing,
                                               score_without_bias[i, :]))
                            all_scores_filter_bias.append((line, missing,
                                                           score[i, :]))
                            line += 1

                    # "argsort" returns the indices that sort a tensor along
                    # a given dimension by value (score)
                    # -> first value = index of entity with highest score
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = pos_sample[:, 0]
                    else:  # mode == 'tail-batch'
                        positive_arg = pos_sample[:, 2]
                    # else:
                    #     raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # positive_arg[i] = index of true entity
                        # rank = position of true entity in models ranking
                        # (lower = better)
                        # nonzero returns indices of elements that are non-zero
                        rank = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert rank.size(0) == 1

                        # rank + 1 is the true rank used in eval. metrics
                        rank = 1 + rank.item()
                        logs.append({
                            'MRR': 1.0/rank,  # higher is better
                            'MR': float(rank),  # lower is better
                            # higher is better for all HITS Scores
                            'HITS@1': 1.0 if rank <= 1 else 0.0,
                            'HITS@3': 1.0 if rank <= 3 else 0.0,
                            'HITS@10': 1.0 if rank <= 10 else 0.0,
                        })
                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)',
                                     step, total_steps)
                    step += 1

        ranks = [int(log['MR']) for log in logs]
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics, all_scores, all_scores_filter_bias, ranks
