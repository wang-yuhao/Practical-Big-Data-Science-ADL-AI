import os
import argparse
import pickle
import torch
import numpy as np
from src.models.api import EvaluationModel, NegSampleGenerator
from torch import nn
from src.models.ensemble.model_selector import ModelSelector, generate_lookup
from src.models.rotate import RotatE
from src.models.conve import ConveEvaluationModel
import mlflow
import mlflow.pytorch

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Load trained model and use it for predictions'
    )
    parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('-d', '--data', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('-p', '--prediction_only', type=bool, default=False)

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

    # get all model structures(wo.parameters yet)
    # Paths might need to be changed, so all models should be in a directory
    # With that we still need only 1 arg
    # Second element is the model name
    models = dict()

    num_samples = 80
    batch_size = 4
    epsilon = 2.0

    hidden_dim = 1000
    rotate_gamma = nn.Parameter(
        torch.Tensor([9.0]),
        requires_grad=False
    )
    embedding_range = nn.Parameter(
        torch.Tensor([(rotate_gamma.item() + epsilon) / hidden_dim]),
        requires_grad=False
    )

    # Loading convE
    mlflow.set_tracking_uri('http://10.195.1.54')
    conve_model = mlflow.pytorch.load_model('sftp://sftpuser@10.195.1.54/sftpuser/mlruns/2/0278ec00cc7b47eda553db7c4f66120e/artifacts/models/conve-model-43') # noqa
    device = torch.device('cuda')
    conve_model.device = device

    tensor_triples = torch.tensor(all_true_triples,
                                  dtype=torch.long,
                                  device=device)
    all_entities = torch.arange(conve_model.num_entities,
                                dtype=torch.long,
                                device=device)
    models['ConvE'] = ConveEvaluationModel(conve_model, tensor_triples,
                                           all_entities, device='cuda')

    models['TransE'] = EvaluationModel(model_class=TransE(transe_gamma),
                                       neg_sample_generator=neg_sample_generator)
    models['RotatE'] = EvaluationModel(model_class=RotatE(embedding_range, rotate_gamma),
                                       neg_sample_generator=neg_sample_generator)
    models['DistMult'] = EvaluationModel(model_class=DistMult(),
                            neg_sample_generator=neg_sample_generator)
    models['RESCAL'] = EvaluationModel(model_class=RESCAL(batch_size=batch_size),
                            neg_sample_generator=neg_sample_generator)



    # Loading DistMult
    distmult_path =  os.path.join(args.model, 'DistMult')
    path = os.path.join(distmult_path, 'entity_embedding.npy')
    new_entity_embedding = nn.Parameter(torch.from_numpy(np.load(path)))
    path = os.path.join(distmult_path, 'relation_embedding.npy')
    new_relation_embedding = nn.Parameter(torch.from_numpy(np.load(path)))

    models['DistMult'].change_entity_embedding(new_entity_embedding.cuda())
    models['DistMult'].change_relation_embedding(new_relation_embedding.cuda())
    models['DistMult'].cuda()
    models['DistMult'].eval()

    # Loading RESCAL
    rescal_path =  os.path.join(args.model, 'RESCAL','checkpoint.pickle')
    checkpoint = torch.load(rescal_path)
    new_entity_embedding = checkpoint['entity_embeddings.weight']
    new_relation_embedding = checkpoint['relation_embeddings.weight']

    models['RESCAL'].change_entity_embedding(new_entity_embedding.cuda())
    models['RESCAL'].change_relation_embedding(new_relation_embedding.cuda())
    models['RESCAL'].cuda()
    models['RESCAL'].eval()

    # Loading TransE
    transe_path =  os.path.join(args.model, 'TransE')
    path = os.path.join(transe_path, 'entity_embedding.npy')
    new_entity_embedding = np.load(path, allow_pickle=True).item().weight
    path = os.path.join(transe_path, 'relation_embedding.npy')
    new_relation_embedding = np.load(path, allow_pickle=True).item().weight

    models['TransE'].change_entity_embedding(new_entity_embedding.cuda())
    models['TransE'].change_relation_embedding(new_relation_embedding.cuda())
    models['TransE'].cuda()
    models['TransE'].eval()

    # Loading RotatE
    rotate_path =  os.path.join(args.model, 'RotatE')
    path = os.path.join(rotate_path, 'entity_embedding.npy')
    new_entity_embedding = nn.Parameter(torch.from_numpy(np.load(path)))
    models['RotatE'].change_entity_embedding(new_entity_embedding.cuda())

    path = os.path.join(rotate_path, 'relation_embedding.npy')
    new_relation_embedding = nn.Parameter(torch.from_numpy(np.load(path)))
    models['RotatE'].change_relation_embedding(new_relation_embedding.cuda())

    models['RotatE'].cuda()
    models['RotatE'].eval()


    lookup_path = os.path.join(args.output_path,
                               'model_selector_lookup.pkl')

    # Calculate lookup for ensemble
    if not args.prediction_only:
        lookup = generate_lookup(models, train_triples, num_samples, batch_size)
        # save lookup
        with open(lookup_path, 'wb') as fp:
            pickle.dump(lookup, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # load lookup
        with open(lookup_path, 'rb') as fp:
            lookup = pickle.load(fp)

    ens_model = ModelSelector(models=models,
                              neg_sample_generator=neg_sample_generator,
                              lookup=lookup)

    s = torch.tensor(test_triples[:, 0]).cuda()
    p = torch.tensor(test_triples[:, 1]).cuda()
    o = torch.tensor(test_triples[:, 2]).cuda()
    evaluation_result = ens_model.evaluate(s, p, o, batch_size=batch_size)

    if args.output_path is not None:
        ens_model.generate_model_output(output_path=args.output_path,
                                    test_triples=test_triples,
                                    evaluation_result=evaluation_result)


if __name__ == '__main__':
    main(parse_args())
