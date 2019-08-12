import logging
import torch
import time
import click
import os

import numpy as np

from os.path import join
from pykeen.utilities import evaluation_utils
from pykeen.kge_models import DistMult
from mlflow import log_metric, log_param
import mlflow


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logging.getLogger('pykeen').setLevel(logging.INFO)


def load_datasets(data='data/processed/FB15k-237/'):
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
        (np.load(join(data, f), allow_pickle=True) for f in mappings)
    test, train, valid = \
        (np.load(join(data, f), allow_pickle=True) for f in datasets)

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


def get_model(num_entities, num_relations, embedding_size, margin_loss):
    '''Initializes a DistMult model.
    Parameters
    ----------
        num_entities - int: The total number of distinct entities in the
                            dataset.
        num_relations - int: The total number of distinct realtions in the
                             dataset.
    '''
    model = DistMult(
        preferred_device='gpu',
        random_seed=1234,
        embedding_dim=embedding_size,
        margin_loss=margin_loss,
        optimizer=torch.optim.Adam
    )

    model.num_entities = num_entities
    model.num_relations = num_relations

    #pylint: disable=protected-access
    model._init_embeddings()

    return model


@click.command()
@click.option('--output', help='Path to output training results', default='model') # noqa
@click.option('--data', help='Path to processes data', default='data/processed/FB15k-237/') #noa
def main(output=None, data=None):
    num_epochs = 300
    learning_rate = 0.1
    batch_size = 2056
    embedding_size = 400
    margin_loss = 10000.0

    preferred_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(preferred_device)

    vocab, datasets = load_datasets(data)

    train, test = datasets['train'], datasets['test']

    id2e, id2rel = vocab['id2e'], vocab['id2rel']

    model = get_model(num_entities=len(id2e), num_relations=len(id2rel),
                      embedding_size=embedding_size, margin_loss=margin_loss)
    model.default_optimizer = torch.optim.Adagrad

    if os.environ.get('DEV'):
        logger.warning('Running in dev mode')

        num_epochs = 1
        train = train[:100]
        test = test[:20]

    mlflow.set_experiment("DistMult")
    with mlflow.start_run(run_name="DistMult"):

        log_param("Epochs", num_epochs)
        log_param("Learning Rate", learning_rate)
        log_param("Batch size", batch_size)
        log_param("Embedding_size", embedding_size)
        log_param("Margin_loss", margin_loss)
        log_param("optimizer", "Adagrad")

        loss_per_epoch = model.fit(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            pos_triples=train,
            tqdm_kwargs=None,
        )

        # %% Save the model and the loss
        timestr = time.strftime("%Y%m%d-%H%M%S")

        torch.save(model.state_dict(), os.path.join(output, f'{timestr}-DistMult.pickle')) # noqa
        torch.save(loss_per_epoch, os.path.join(output, f'{timestr}-DistMult-loss.pickle')) # noqa

        metrics = evaluation_utils.metrics_computations.compute_metric_results(
            kg_embedding_model=model,
            mapped_train_triples=train,
            mapped_test_triples=test,
            device=device
        )

        print(metrics.mean_rank)
        print(metrics.mean_reciprocal_rank)
        print(metrics.hits_at_k)

        log_metric("Mean Rank", metrics.mean_rank)
        log_metric("Mean Reciprocal Rank", metrics.mean_reciprocal_rank)

        for k, v in metrics.hits_at_k.items():
            log_metric(f'Hits_at_{k}', v)


if __name__ == "__main__":
    main()
