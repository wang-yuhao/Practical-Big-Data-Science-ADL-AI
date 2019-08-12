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


class DistMult:

    def get_score(self, head: torch.tensor, relation: torch.tensor,
                  tail: torch.tensor, mode: str) -> torch.tensor:
        """
        Computes Scores for head, relation, tail triples with DistMult model

        :param head: torch.tensor, dtype: int, shape: (batch_size, sample_size,
        entity_dim)
        :param relation: torch.tensor, dtype: int, shape: (batch_size,
        sample_size, relation_dim)
        :param tail: torch.tensor, dtype: int, shape: (batch_size, sample_size,
        entity_dim)
        :param mode: str ('single', 'head-batch' or 'head-tail')

        :return: torch.tensor, dtype: float, shape: (batch_size, num_entities)
        """

        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Load trained model and use it for predictions'
    )
    parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('-d', '--data', type=str, default=None)
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

    # load data
    train_triples, valid_triples, test_triples = load_data(args.data)

    # create model and load already trained embeddings
    all_true_triples = np.concatenate([train_triples, valid_triples,
                                       test_triples], axis=0)
    neg_sample_generator = NegSampleGenerator(all_true_triples,
                                              create_filter_bias=True)
    model = EvaluationModel(model_class=DistMult(),
                            neg_sample_generator=neg_sample_generator)

    path = os.path.join(args.model, 'entity_embedding.npy')
    new_entity_embedding = nn.Parameter(torch.from_numpy(np.load(path)))

    path = os.path.join(args.model, 'relation_embedding.npy')
    new_relation_embedding = nn.Parameter(torch.from_numpy(np.load(path)))

    # only True if embeddings of RotatE are used
    if new_entity_embedding.shape[1] != new_relation_embedding.shape[1]:
        stop = new_relation_embedding.shape[1]
        new_entity_embedding = new_entity_embedding[:, :stop]
        new_entity_embedding = nn.Parameter(new_entity_embedding)

    model.change_entity_embedding(new_entity_embedding.cuda())
    model.change_relation_embedding(new_relation_embedding.cuda())

    model.cuda()
    model.eval()

    # use API to evaluate model and generate model output for error analysis
    s = torch.tensor(test_triples[:, 0]).cuda()
    p = torch.tensor(test_triples[:, 1]).cuda()
    o = torch.tensor(test_triples[:, 2]).cuda()
    evaluation_result = model.evaluate(s, p, o, batch_size=4)

    if args.output_path is not None:
        model.generate_model_output(output_path=args.output_path,
                                    test_triples=test_triples,
                                    evaluation_result=evaluation_result)


if __name__ == '__main__':
    main(parse_args())
