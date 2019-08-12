import logging
import torch
import time
import click
import os
import argparse

import numpy as np

from os.path import join
from pykeen.utilities import evaluation_utils
from pykeen.kge_models import RESCAL
from mlflow import log_metric, log_param
import mlflow


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logging.getLogger('pykeen').setLevel(logging.INFO)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='run_rescal.py [<args>] [-h | --help]'
    )
    parser.add_argument('-e', '--num_epochs', default=500, type=float, help='The number of epochs')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float, help='The learning rate in training model')
    parser.add_argument('-b', '--batch_size', default=100, type=float, help="The batch size ")
    parser.add_argument('-p','--preferred_device', default='gpu', type=str,help="The preferred device,only 'gpu' or 'cpu'")
    parser.add_argument('-r', '--random_seed', default=0, type=float, help="The random seed")
    parser.add_argument('-dim', '--embedding_dim', default=100, type=float,help='The dimensions of embeddings')
    parser.add_argument('-s', '--scoring_function', default=2, type=float, help='Scoring function of Rescal')
    parser.add_argument('-g', '--margin_loss', default=1, type=float, help='The margin of loss function')
    parser.add_argument('-opt', '--optimizer', default='ADAGRAD', type=str, help='The optimizer for update weights')
    parser.add_argument('-d','--data_path', default='../data/processed/FB15k-237/',type=str, help='The directory of the train, test and validate dataset')
    parser.add_argument('-o','--output', default='../models',type=str,  help='The directory to save models and trained embeddings')

    return parser.parse_args(args)


def load_datasets(data_path):
    '''Reads in the index mappings and dataset of the FB15k-237 dataset.
    Returns
    -------
        vocab - Dict: The index mapping for the FB15k-237 dataset with the
                      following mappings:
                      e2id --> entity to index
                      id2e --> index to entity
                      id2rel --> index to relation
                      rel2id --> relation to index
        dataset - Dict: A dictionary containing the train, valid and
                        test dataset.
    '''
    base_path = join(data_path)

    mappings = [
        'entity_to_id.pickle',
        'id_to_entity.pickle',
        'id_to_relation.pickle',
        'relation_to_id.pickle'
    ]

    datasets = [
        'test.pickle',
        'train.pickle',
        'valid.pickle'
    ]

    e2id, id2e, id2rel, rel2id = \
        (np.load(join(base_path, f), allow_pickle=True) for f in mappings)
    test, train, valid = \
        (np.load(join(base_path, f), allow_pickle=True) for f in datasets)

    vocab = dict(
        e2id=e2id,
        id2e=id2e,
        id2rel=id2rel,
        rel2id=rel2id
    )

    datasets = dict(
        train=train,
        valid=valid,
        test=test
    )
    return vocab, datasets


def get_model(args, num_entities, num_relations):
    '''Initializes a RESCAL model.
    Parameters
    ----------
        num_entities - int: The total number of distinct entities in the
                            dataset.
        num_relations - int: The total number of distinct realtions in the
                             dataset.
    '''
    model = RESCAL(
        preferred_device=args.preferred_device,
        random_seed=args.random_seed,
        embedding_dim=args.embedding_dim,
        margin_loss=args.margin_loss,
        scoring_function=args.scoring_function,
    )

    model.num_entities = num_entities
    model.num_relations = num_relations

    # model._init_embeddings()

    return model

#@click.command()
#@click.option('--output', help='Path to output training results', default='model') # noqa
#@click.option('--data', help='Path to processes data', default='data/processed/FB15k-237/') #noa
def main(args):
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim
    scoring_function = args.scoring_function
    optimizer = args.optimizer

    preferred_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(preferred_device)

    vocab, datasets = load_datasets(args.data_path)

    train, test = datasets['train'], datasets['test']

    id2e, id2rel = vocab['id2e'], vocab['id2rel']

    model = get_model(args, num_entities=len(id2e), num_relations=len(id2rel))

    if os.environ.get('DEV'):
        logger.warning('Running in dev mode')

        num_epochs = 1
        train = train[:100]
        test = test[:20]

    mlflow.set_experiment("RESCAL")
    with mlflow.start_run():

        log_param("Epochs", num_epochs)
        log_param("Learning Rate", learning_rate)
        log_param("Batch size", batch_size)
        log_param("Embedding Dimension", embedding_dim)
        log_param("Scoring Function", scoring_function)
        log_param("Optimizer", optimizer)

        loss_per_epoch = model.fit(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            pos_triples=train,
            tqdm_kwargs=None,
            optimizer=torch.optim.Adagrad
        )

        # %% Save the model and the loss
        timestr = time.strftime("%Y%m%d-%H%M%S")

        torch.save(
            model.state_dict(), os.path.join(output, f'{timestr}-RESCAL.pickle')
        )

        torch.save(
            loss_per_epoch, os.path.join(output, f'{timestr}-RESCAL-loss.pickle')
        )

        entity_embedding = model.entity_embeddings
        np.save(
            os.path.join(output, 'entity_embedding'),
            entity_embedding
        )

        relation_embedding = model.relation_embeddings
        np.save(
            os.path.join(output, 'relation_embedding'),
            relation_embedding
        )

        metrics = evaluation_utils.metrics_computations.compute_metric_results(
            kg_embedding_model=model,
            mapped_train_triples=train,
            mapped_test_triples=test,
            device=device
        )

        log_metric("Mean Rank", metrics.mean_rank)
        log_metric("Mean Reciprocal Rank", metrics.mean_reciprocal_rank)

        for k, v in metrics.hits_at_k.items():
            log_metric(f'Hits_at_{k}', v)


if __name__ == "__main__":
    main(parse_args())
