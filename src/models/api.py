import abc
import random
from collections import defaultdict
from typing import Optional

from torch import nn
import numpy as np
import torch
import tqdm

from src.models.generate_model_output import OutputGenerator


class AbstractModel(nn.Module, abc.ABC):
    def __init__(self, num_entities: int, num_relations: int):
        super(AbstractModel, self).__init__()
        self.num_entities = num_entities
        self.num_relation = num_relations

    def score_subjects(self, p: torch.tensor, o: torch.tensor) -> torch.tensor:
        """
        Compute subject scores for all entities, given p and o.

        :param p: torch.tensor, scalar
            The predicate.
        :param o: torch.tensor, scalar
            The object.

        :return: torch.tensor, shape: (num_entities,)
            The scores for all entities.
        """
        raise NotImplementedError()

    def score_objects(self, s: torch.tensor, p: torch.tensor) -> torch.tensor:
        """
        Compute subject scores for all entities, given p and o.

        :param s: torch.tensor, scalar
            The subject.
        :param p: torch.tensor, scalar
            The predicate.

        :return: torch.tensor, shape: (num_entities,)
            The scores for all entities.
        """
        raise NotImplementedError()


def _compute_rank_from_scores(
        scores: torch.tensor,
        true_idx: torch.tensor,
        mask: torch.tensor
) -> np.ndarray:
    """
    Given scores, computes the rank.

    :param scores: torch.tensor, shape: (num_entities,)
        The array of scores for all entities.
    :param true_idx: torch.tensor, scalar, int
        The idx of the true element.
    :param mask: torch.tensor, bool
        A mask to exclude elements from ranking (e.g. for filtered setting).

    :return: np.ndarray, scalar, float
        The rank.
    """
    true_score = scores[true_idx]
    other_scores = scores[mask]
    # How many scores are better?
    best_rank = torch.sum(other_scores > true_score)
    # How many scores are not worse?
    worst_rank = torch.sum(other_scores >= true_score)
    rank = (best_rank + worst_rank) / 2.0 + 1
    rank = rank.detach().cpu().numpy()
    return rank


def evaluate(
        triples: np.ndarray,
        model: AbstractModel,
        device: torch.device = torch.device('cuda:0'),
        ks=(1, 3, 10),
) -> np.ndarray:
    """
    Given a set of test triples, computes several ranking-based metrics.

    :param triples: numpy.ndarray, dtype: int, shape: (n_triples, 3)
        The triples in format (s, p, o)
    :param model: AbstractModel
        The model to evaluate.
    :param device: torch.device
        The device to use for evaluation.
    :param ks: Tuple[int]
        The values for k for which Hits@k is computed.

    :return: numpy.ndarray, dtype: float, shape (n_scores,)
        The scores:
            out[0] = mean_rank
            out[0] = mean_reciprocal_rank
            out[2:] = hits@k
    """
    n_triples = triples.shape[0]

    # compute subject-, and object-based ranks
    ranks = np.empty(shape=(n_triples, 2), dtype=np.float)

    # Do not track any gradients
    with torch.no_grad():
        # Send model to device
        model = model.to(device)

        # Send triples to device
        triples = torch.tensor(triples, device=device, dtype=torch.long)

        all_entities = torch.arange(
            model.num_entities,
            dtype=torch.long,
            device=device
        )
        for i in tqdm.trange(n_triples, unit='triple', unit_scale=True):
            # Split triple
            s, p, o = triples[i, :]

            # Left-side link prediction
            subject_mask = all_entities != s
            subject_scores = model.score_subjects(p, o)
            ranks[i, 0] = _compute_rank_from_scores(
                scores=subject_scores,
                true_idx=s,
                mask=subject_mask
            )

            # Right-side link prediction
            object_mask = all_entities != o
            object_scores = model.score_objects(s, p)
            ranks[i, 1] = _compute_rank_from_scores(
                scores=object_scores,
                true_idx=o,
                mask=object_mask
            )

    # Compute scores
    mean_rank = np.mean(ranks)
    mean_reciprocal_rank = np.mean(np.reciprocal(ranks))
    hits = [np.mean(ranks <= k) for k in ks]
    scores = np.asarray([mean_rank, mean_reciprocal_rank] + hits)
    return scores


class EvaluationModel(nn.Module, abc.ABC):
    def __init__(self, model_class, neg_sample_generator):

        super(EvaluationModel, self).__init__()
        self.model_class = model_class

        self.neg_sample_generator = neg_sample_generator

        #: The number of entities in the knowledge graph
        self.num_entities = 0

        #: The number of unique relation types in the knowledge graph
        self.num_relations = 0

        self.entity_embedding = None
        self.relation_embedding = None

    def init_embeddings(self, num_entities: int, num_relations: int,
                        embedding_range: nn.Parameter) -> None:
        """
        Initialise the embeddings (to be done by subclass)

        :param num_entities: int, > 0
            The number of unique entities
        :param num_relations: int, > 0
            The number of unique relations

        :return: None
        """
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.entity_embedding = nn.Parameter(torch.zeros(num_entities,
                                             self.embedding_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relations,
                                                           self.embedding_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )

        # alternative:
        # model_path = os.path.join(args.model, 'checkpoint')
        # checkpoint = torch.load(model_path)
        # model.load_state_dict(checkpoint['model_state_dict'])

    def change_entity_embedding(self, new_embedding: nn.Parameter) -> None:
        """
        Change the entity embeddings (to be done by subclass)

        :param new_embedding: torch.nn.parameter, dtype: float,
                              shape: (num_entities, entity_dim)

        :return: None
        """
        self.entity_embedding = new_embedding
        self.num_entities = new_embedding.size(0)

    def change_relation_embedding(self, new_embedding: nn.Parameter) -> None:
        """
        Change the relation embeddings (to be done by subclass)

        :param new_embedding: torch.nn.parameter, dtype: float,
                              shape: (num_relations, relation_dim)

        :return: None
        """
        self.relation_embedding = new_embedding
        self.num_relations = new_embedding.size(0)

    def compute_metrics(self, ranks: list) -> dict:
        """
        Compute MRR, MR, HITS@1, HITS@3, HITS@10 for the given ranks

        :param ranks: list, dtype: int
            rank ist the position of true entity in models ranking

        :return: dict
            keys: MRR, MR, HITS@1, HITS@3, HITS@10
            values: corresponding value
        """
        logs = []
        for rank in ranks:
            logs.append({
                'MRR': 1.0/rank,
                'MR': float(rank),
                'HITS@1': 1.0 if rank <= 1 else 0.0,
                'HITS@3': 1.0 if rank <= 3 else 0.0,
                'HITS@10': 1.0 if rank <= 10 else 0.0,
            })
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        return metrics

    def evaluate_only_metrics(self, s: torch.tensor, p: torch.tensor,
                              o: torch.tensor, batch_size) -> dict:
        """
        Evaluation of given triples look at every triple twice (predict head
        and predict tail), compute metrics based on ranks

        :param s: torch.tensor, dtype: int, shape: (dataset_size,)
            The subjects' IDs.
        :param p: torch.tensor, dtype: int, shape: (dataset_size,)
            The predicates' IDs
        :param o: torch.tensor, dtype: int, shape: (dataset_size,)
            The objects' IDs
        :param batch_size: int

        :return: metrics (dictionary)
        """
        assert len(s) == len(p) == len(o)
        ranks = []

        for i in range(0, len(s), batch_size):
            scores_object = self.predict_object_scores(s[i:i+batch_size],
                                                       p[i:i+batch_size],
                                                       o[i:i+batch_size])

            argsort_object = torch.argsort(scores_object, dim=1,
                                           descending=True)

            scores_subject = self.predict_subject_scores(s[i:i+batch_size],
                                                         p[i:i+batch_size],
                                                         o[i:i+batch_size])
            argsort_subject = torch.argsort(scores_subject, dim=1,
                                            descending=True)

            positive_arg_object = o[i:i+batch_size]
            positive_arg_subject = s[i:i+batch_size]

            last_iteration = False
            for b in range(batch_size):
                if b == len(positive_arg_object)-1:
                    last_iteration = True

                rank_object = (argsort_object[b, :] ==
                               positive_arg_object[b]).nonzero()
                assert rank_object.size(0) == 1
                # rank + 1 is the true rank used in eval. metrics
                rank_object = 1 + rank_object.item()

                rank_subject = (argsort_subject[b, :] ==
                                positive_arg_subject[b]).nonzero()
                assert rank_subject.size(0) == 1
                # rank + 1 is the true rank used in eval. metrics
                rank_subject = 1 + rank_subject.item()

                ranks.append(float(rank_object))
                ranks.append(float(rank_subject))

                if last_iteration:
                    break

        assert len(s) == int(len(ranks)/2)
        return self.compute_metrics(ranks)

    def evaluate(self, s: torch.tensor, p: torch.tensor, o: torch.tensor,  # noqa
                 batch_size) -> dict:
        """
        Evaluation of given triples look at every triple twice (predict head
        and predict tail), creates output that can be useful for error analysis
        and model comparison

        :param s: torch.tensor, dtype: int, shape: (dataset_size,)
            The subjects' IDs.
        :param p: torch.tensor, dtype: int, shape: (dataset_size,)
            The predicates' IDs
        :param o: torch.tensor, dtype: int, shape: (dataset_size,)
            The objects' IDs
        :param batch_size: int

        :return: ranks, score_examples, highly_ranked_objects_count,
        highly_ranked_subjects_count, top_ranked_objects_count,
        top_ranked_subjects_count, average_position_objects,
        average_position_subjects
        """
        assert len(s) == len(p) == len(o)
        ranks = []
        random_index = random.randint(0, len(s)-1)
        score_examples = []
        average_position_objects = defaultdict(int)
        average_position_subjects = defaultdict(int)
        highly_ranked_objects_count = defaultdict(int)  # within top10
        highly_ranked_subjects_count = defaultdict(int)  # within top10
        top_ranked_objects_count = defaultdict(int)  # on first place
        top_ranked_subjects_count = defaultdict(int)  # on first place

        for i in tqdm.tqdm(range(0, len(s), batch_size)):
            scores_object = self.predict_object_scores(s[i:i+batch_size],
                                                       p[i:i+batch_size],
                                                       o[i:i+batch_size])
            argsort_object = torch.argsort(scores_object, dim=1,
                                           descending=True)

            scores_subject = self.predict_subject_scores(s[i:i+batch_size],
                                                         p[i:i+batch_size],
                                                         o[i:i+batch_size])
            argsort_subject = torch.argsort(scores_subject, dim=1,
                                            descending=True)

            assert argsort_object.shape == argsort_subject.shape
            last_iteration = False
            for b in range(batch_size):
                if b == len(argsort_object)-1:
                    last_iteration = True
                position = 0
                while position < len(argsort_object[b]):
                    index_object = argsort_object[b][position].item()
                    average_position_objects[index_object] += position

                    index_subject = argsort_subject[b][position].item()
                    average_position_subjects[index_subject] += position

                    if position < 10:  # within top10
                        highly_ranked_subjects_count[index_subject] += 1
                        highly_ranked_objects_count[index_object] += 1
                        if position == 0:
                            top_ranked_subjects_count[index_subject] += 1
                            top_ranked_objects_count[index_object] += 1

                    position += 1

                if last_iteration:
                    break

            positive_arg_object = o[i:i+batch_size]
            positive_arg_subject = s[i:i+batch_size]

            last_iteration = False
            for b in range(batch_size):
                if b == len(positive_arg_object)-1:
                    last_iteration = True
                if i + b == random_index:
                    score_examples.append(scores_object[b, :])
                    score_examples.append(scores_subject[b, :])
                rank_object = (argsort_object[b, :] ==
                               positive_arg_object[b]).nonzero()
                assert rank_object.size(0) == 1
                # rank + 1 is the true rank used in eval. metrics
                rank_object = 1 + rank_object.item()

                rank_subject = (argsort_subject[b, :] ==
                                positive_arg_subject[b]).nonzero()
                assert rank_subject.size(0) == 1
                # rank + 1 is the true rank used in eval. metrics
                rank_subject = 1 + rank_subject.item()

                ranks.append(float(rank_object))
                ranks.append(float(rank_subject))

                if last_iteration:
                    break

        assert len(s) == int(len(ranks)/2)
        average_position_objects = {key: value/len(s) for key, value in
                                    average_position_objects.items()}
        average_position_subjects = {key: value/len(s) for key, value in
                                     average_position_subjects.items()}

        evaluation_result = [ranks, score_examples]
        evaluation_result.append(highly_ranked_objects_count)
        evaluation_result.append(highly_ranked_subjects_count)
        evaluation_result.append(top_ranked_objects_count)
        evaluation_result.append(top_ranked_subjects_count)
        evaluation_result.append(average_position_objects)
        evaluation_result.append(average_position_subjects)
        return evaluation_result

    def predict_triple_scores(self, s: torch.tensor, p: torch.tensor,
                              o: torch.tensor) -> torch.tensor:
        """
        Fact prediction.

        :param s: torch.tensor, dtype: int, shape: (batch_size,)
            The subjects' IDs.
        :param p: torch.tensor, dtype: int, shape: (batch_size,)
            The predicates' IDs
        :param o: torch.tensor, dtype: int, shape: (batch_size,)
            The objects' IDs.

        :return: torch.tensor, dtype: float, shape: (batch_size,)
            The scores for each triple.
        """

        assert len(s) == len(p) == len(o)

        subjects = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=s
        ).unsqueeze(1)

        predicates = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=p
        ).unsqueeze(1)

        objects = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=o
        ).unsqueeze(1)

        score = self.model_class.get_score(subjects, predicates, objects,
                                           mode='single')
        return score

    def predict_object_scores(self, s: torch.tensor, p: torch.tensor,
                              o: torch.tensor) -> torch.tensor:
        """
        Link Prediction (right-sided)

        :param s: torch.tensor, dtype: int, shape: (batch_size,)
            The subjects' IDs.
        :param p: torch.tensor, dtype: int, shape: (batch_size,)
            The predicates' IDs
        :param o: torch.tensor, dtype: int, shape: (batch_size,)
            The objects' IDs.

        :return: torch.tensor, dtype: float, shape: (batch_size, num_entities)
        """
        assert len(s) == len(p)
        subjects = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=s
        ).unsqueeze(1)

        predicates = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=p
        ).unsqueeze(1)

        batch_size = len(s)
        neg_sample, filter_bias = self.neg_sample_generator.\
        get_neg_sample(s, p, o, self.num_entities, mode='tail-batch')  # noqa

        # neg_sample = [tail for tail in range(self.num_entities)]
        # neg_sample = torch.tensor(np.array([neg_sample] * batch_size)).to(device=device) # noqa
        neg_sample_size = neg_sample.size(1)
        objects = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=neg_sample.view(-1)
        ).view(batch_size, neg_sample_size, -1)

        score = self.model_class.get_score(subjects, predicates, objects,
                                           mode='tail-batch')

        assert score.shape == filter_bias.shape
        return score + filter_bias

    def predict_subject_scores(self, s: torch.tensor, p: torch.tensor,
                               o: torch.tensor) -> torch.tensor:
        """
        Link Prediction (left-sided)

        :param s: torch.tensor, dtype: int, shape: (batch_size,)
            The subjects' IDs.
        :param p: torch.tensor, dtype: int, shape: (batch_size,)
            The predicates' IDs
        :param o: torch.tensor, dtype: int, shape: (batch_size,)
            The objects' IDs.

        :return: torch.tensor, dtype: float, shape: (batch_size, num_entities)
        """
        assert len(p) == len(o)

        predicates = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=p
        ).unsqueeze(1)

        objects = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=o
        ).unsqueeze(1)

        batch_size = len(o)
        neg_sample, filter_bias = self.neg_sample_generator.\
        get_neg_sample(s, p, o, self.num_entities, mode='head-batch')  # noqa
        # neg_sample = [head for head in range(self.num_entities)]
        # neg_sample = torch.tensor(np.array([neg_sample] * batch_size)).to(device=device) # noqa
        neg_sample_size = neg_sample.size(1)

        subjects = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=neg_sample.view(-1)
        ).view(batch_size, neg_sample_size, -1)

        score = self.model_class.get_score(subjects, predicates, objects,
                                           mode='head-batch')
        assert score.shape == filter_bias.shape
        return score + filter_bias

    def generate_model_output(self, output_path: str,
                              test_triples: np.ndarray,
                              evaluation_result: list) -> None:
        """
        Generates the model output needed for Error Analysis and comparison
        of different models using the OutputGenerator class

        :param output_path: path to the folder, where everything will be saved
        :param test_triples: np.ndarray, dtype: int, shape: (num_samples, 3)
        :param evaluation_result: list (ranks, score_examples,
        highly_ranked_objects_count, highly_ranked_subjects_count,
        top_ranked_objects_count, top_ranked_subjects_count,
        average_position_objects, average_position_subjects)
)

        :return: None
        """

        ranks = evaluation_result[0]
        score_examples = evaluation_result[1]
        highly_ranked_objects_count = evaluation_result[2]
        highly_ranked_subjects_count = evaluation_result[3]
        top_ranked_objects_count = evaluation_result[4]
        top_ranked_subjects_count = evaluation_result[5]
        average_position_objects = evaluation_result[6]
        average_position_subjects = evaluation_result[7]

        output_generator = OutputGenerator(output_path)

        # create tables with rank per triple
        output_generator.create_prediction_tables(test_triples, ranks)

        # create bar chart and CDF plot of scores (for one random prediction)
        for example in score_examples:
            output_generator.plot_scores_of_one_prediction(example)

        # find examples of good and bad predictions
        output_generator.find_prediction_examples(test_triples, ranks,
                                                  good=True)
        output_generator.find_prediction_examples(test_triples, ranks,
                                                  good=False)

        # compute metrics (MR, MRR and H@N) for head / tail separately
        output_generator.compare_head_tail_predictions(ranks)

        # create tables of the average position of an entitiy
        output_generator.create_average_position_table(average_position_subjects,  # noqa
                                                       average_position_objects)  # noqa

        # create tables of the count: how often the respective
        # entity gets a position between 1 and 10
        output_generator.create_highly_ranked_count_tables(highly_ranked_subjects_count,  # noqa
                                                           highly_ranked_objects_count)  # noqa
        output_generator.create_top_ranked_count_tables(top_ranked_subjects_count,  # noqa
                                                        top_ranked_objects_count)  # noqa
        # output_generator.log_top10(test_triples, scores)

    #pylint: disable=arguments-differ
    def forward(self, pos_batch, mode='single'):
        s = pos_batch[:, 0]
        p = pos_batch[:, 1]
        o = pos_batch[:, 2]
        if mode == 'single':
            return self.predict_triple_scores(s=s, p=p, o=o)
        elif mode == 'head-batch':
            return self.predict_subject_scores(s=s, p=p, o=o)
        else:  # tail-batch
            return self.predict_object_scores(s=s, p=p, o=o)


class NegSampleGenerator:
    def __init__(self, all_true_triples: np.ndarray,
                 create_filter_bias: Optional[bool] = False,
                 filter_bias: Optional[torch.tensor] = None,
                 bias: Optional[int] = -1):
        self.triple_set = set(tuple(i) for i in all_true_triples)
        self.create_filter_bias = create_filter_bias
        self.filter_bias = filter_bias
        self.bias = bias
        if self.create_filter_bias and self.filter_bias is not None:
            raise ValueError('create_filter_bias has to be False or'
                             + ' filter bias has to be None')

    def get_neg_sample(self,
                       head: torch.tensor,
                       relation: torch.tensor,
                       tail: torch.tensor,
                       num_entities: int,
                       mode: str,
                       device: str = 'cuda'
                       ) -> tuple:
        """
        Get negative sample and filter_bias (only if create_filter_bias = True
        or filter_bias is not None) for a given triple

        :param head: torch.tensor, dtype: int, shape: (batch_size,)
            The subjects' IDs.
        :param relation: torch.tensor, dtype: int, shape: (batch_size,)
            The predicates' IDs
        :param tail: torch.tensor, dtype: int, shape: (batch_size,)
            The objects' IDs.
        :param num_entities: int, > 0
            The number of unique entities
        :param mode: 'head-batch' or 'head-tail'

        :return: tuple (torch.tensor, torch.tensor)
            neg_sample and optionally filter_bias,
            respective shape: (batch_size, num_entitites)
        """
        device = torch.device(device) \
            if torch.cuda.is_available() \
            else torch.device('cpu')

        all_neg_samples = []
        all_filter_bias = []
        for i in range(len(relation)):

            if mode == 'head-batch':
                tmp = [(0, rand_head) if (rand_head, relation[i].item(),
                                          tail[i].item())
                       not in self.triple_set
                       else (self.bias, head[i]) for rand_head in range(num_entities)]  # noqa
                tmp[head[i]] = (0, head[i])
            elif mode == 'tail-batch':
                tmp = [(0, rand_tail) if (head[i].item(), relation[i].item(),
                                          rand_tail)
                       not in self.triple_set
                       else (self.bias, tail[i]) for rand_tail in range(num_entities)]  # noqa
                tmp[tail[i]] = (0, tail[i])
            else:
                raise ValueError('negative batch mode %s not supported' % mode)

            tmp = torch.LongTensor(tmp)
            if self.create_filter_bias:
                filter_bias = tmp[:, 0].float()
                all_filter_bias.append(filter_bias)
            neg_sample = tmp[:, 1]
            all_neg_samples.append(neg_sample)

        neg_samples = torch.stack(all_neg_samples).to(device=device)
        if self.create_filter_bias:
            filter_bias = torch.stack(all_filter_bias).to(device=device)
            return neg_samples, filter_bias
        elif self.filter_bias is not None:
            return neg_samples, self.filter_bias
        else:
            filter_bias = torch.zeros(neg_samples.shape[0], num_entities)
            return neg_samples, filter_bias
