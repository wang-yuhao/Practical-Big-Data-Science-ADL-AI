import argparse
import pickle
import torch
import os
import numpy as np
from src.models.api import EvaluationModel, NegSampleGenerator
from torch import nn


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class RotatE:
    def __init__(self, embedding_range, gamma):
        self.embedding_range = embedding_range
        self.gamma = gamma

    def get_score(self, head: torch.tensor, relation: torch.tensor,
                  tail: torch.tensor, mode: str) -> torch.tensor:
        """
        Computes Scores for head, relation, tail triples with the RotatE model

        :param head: torch.tensor, dtype: int, shape: (batch_size, sample_size,
        entity_dim)
        :param relation: torch.tensor, dtype: int, shape: (batch_size,
        sample_size, relation_dim)
        :param tail: torch.tensor, dtype: int, shape: (batch_size, sample_size,
        entity_dim)
        :param mode: str ('single', 'head-batch' or 'head-tail')

        :return: torch.tensor, dtype: float, shape: (batch_size, num_entities)
        """

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


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Load trained model and use it for predictions'
    )
    parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('-d', '--data', type=str, default=None)
    parser.add_argument('-dim', '--hidden_dim', type=int, default=1000)
    parser.add_argument('-o', '--output_path', type=str, default=None)

    return parser.parse_args(args)


def load_data(data_path):
    path_train = os.path.join(data_path, 'train.pickle')
    with open(path_train, 'rb') as handle:
        train = pickle.load(handle)

    path_valid = os.path.join(data_path, 'valid.pickle')
    with open(path_valid, 'rb') as handle:
        valid = pickle.load(handle)

    path_test = os.path.join(data_path, 'test.pickle')
    with open(path_test, 'rb') as handle:
        test = pickle.load(handle)

    return train, valid, test


def main(args):
    """
    Load trained model and use it for predictions.
    """

    if args.model is None or args.data is None:
        raise ValueError('You have to specify model and data input paths.')

    epsilon = 2.0
    gamma = nn.Parameter(
        torch.Tensor([9.0]),
        requires_grad=False
    )
    embedding_range = nn.Parameter(
        torch.Tensor([(gamma.item() + epsilon) / args.hidden_dim]),
        requires_grad=False
    )

    # load data
    train_triples, valid_triples, test_triples = load_data(args.data)

    # create model and load already trained embeddings
    all_true_triples = np.concatenate([train_triples, valid_triples,
                                       test_triples], axis=0)
    neg_sample_generator = NegSampleGenerator(all_true_triples,
                                              create_filter_bias=True,
                                              bias=-1000)
    model = EvaluationModel(model_class=RotatE(embedding_range, gamma),
                            neg_sample_generator=neg_sample_generator)
    path = os.path.join(args.model, 'entity_embedding.npy')
    new_entity_embedding = nn.Parameter(torch.from_numpy(np.load(path)))
    model.change_entity_embedding(new_entity_embedding.cuda())

    path = os.path.join(args.model, 'relation_embedding.npy')
    new_relation_embedding = nn.Parameter(torch.from_numpy(np.load(path)))
    model.change_relation_embedding(new_relation_embedding.cuda())

    model.cuda()
    model.eval()
    # use API to evaluate model and generate model output for error analysis
    s = torch.tensor(test_triples[:, 0]).cuda()
    p = torch.tensor(test_triples[:, 1]).cuda()
    o = torch.tensor(test_triples[:, 2]).cuda()

    # print(model.evaluate_only_metrics(s, p, o, batch_size=4))
    evaluation_result = model.evaluate(s, p, o, batch_size=4)
    if args.output_path is not None:
        model.generate_model_output(output_path=args.output_path,
                                    test_triples=test_triples,
                                    evaluation_result=evaluation_result)


if __name__ == '__main__':
    main(parse_args())
