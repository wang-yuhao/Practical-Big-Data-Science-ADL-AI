import os # noqa
import sys # noqa

sys.path.append(os.path.abspath('.')) # noqa
sys.path.append(os.path.abspath('./src/models')) # noqa

import torch
import click
import mlflow
import mlflow.pytorch

import numpy as np

from src.models.ensemble.base import EnsembleModel
from src.models.ensemble.units import TranseUnit, DistMultUnit
from src.data.utils import load


HELP_RUN_NAME = f"If run with provided name already exists for the mlflow experiment \
                  continues to log data to that run, otherwise creates new run."

PARAM_DATA_PATH = './data/processed/FB15k-237/'
HELP_DATA_PATH = f"Path to where the FB15k-237 data is located.\
                   (default: {PARAM_DATA_PATH})"

PARAM_EPOCHS = 50
HELP_EPOCHS = f"Sets the number of epochs to train the ensemble model.\
                (default: {PARAM_EPOCHS})"

PARAM_DEVICE = 'cpu'
HELP_DEVICE = f"Either cpu or cuda. (defaults: {PARAM_DEVICE}"

PARAM_TRACK_URI = 'http://10.195.1.54'
HELP_TRACK_URI = f"The uri pointing to the mlflow tracking server.\
                   Set this option to `file:/my/local/dir` to store\
                   artifacts locally.\
                   (default: {PARAM_TRACK_URI})"

PARAM_EXPERIMENT = 'ensemble-model'
HELP_EXPERIMENT = f"The experiment name to log the data to.\
                    (defaut: ensemble-model)"


def initialize_units(preferred_device='cpu'):
    dismult_entity_embed = np.load(
        'models/trained_models/DistMult/entity_embedding.npy',
        allow_pickle=True)

    dismult_relation_embed = np.load(
        'models/trained_models/DistMult/relation_embedding.npy',
        allow_pickle=True)

    entity_embedding = torch.Tensor(dismult_entity_embed)
    entity_embedding = entity_embedding.to(device=preferred_device)
    relation_embedding = torch.Tensor(dismult_relation_embed)
    relation_embedding = relation_embedding.to(device=preferred_device)

    distmult = DistMultUnit(num_relations=dismult_relation_embed.shape[0],
                            num_entities=dismult_entity_embed.shape[0],
                            entity_embeddings=entity_embedding,
                            relation_embeddings=relation_embedding,
                            preferred_device=preferred_device)

    transe_entity_embed = np.load(
        'models/trained_models/TransE/entity_embedding_new.npy',
        allow_pickle=True)
    transe_relation_embed = np.load(
        'models/trained_models/TransE/relation_embedding_new.npy',
        allow_pickle=True)

    entity_embedding = torch.Tensor(transe_entity_embed)
    entity_embedding = entity_embedding.to(device=preferred_device)
    relation_embedding = torch.Tensor(transe_relation_embed)
    relation_embedding = relation_embedding.to(device=preferred_device)

    transe = TranseUnit(num_relations=transe_relation_embed.shape[0],
                        num_entities=transe_entity_embed.shape[0],
                        entity_embeddings=entity_embedding,
                        relation_embeddings=relation_embedding,
                        preferred_device=preferred_device)

    return [distmult, transe]


@click.command()
@click.option('-n', '--mlflow_run_name', required=True, help=HELP_RUN_NAME)
@click.option('--preferred_device', default=PARAM_DEVICE, help=HELP_DEVICE)
@click.option('--mlflow_tracking_uri', default=PARAM_TRACK_URI, help=HELP_TRACK_URI)
@click.option('--mlflow_experiment_name', default=PARAM_EXPERIMENT, help=HELP_EXPERIMENT)
@click.option('--epochs', default=PARAM_EPOCHS, help=HELP_EPOCHS)
@click.option('--data_path', default=PARAM_DATA_PATH, HELP=HELP_DATA_PATH)
def main(mlflow_run_name=None, preferred_device=None, mlflow_tracking_uri=None,
         mlflow_experiment_name=None, epochs=None, data_path=None):
    """Trains an Ensemble model containing: (DistMult, TransE)."""

    maps, data = load(data_path)

    units = initialize_units(preferred_device)

    model = EnsembleModel(units,
                          num_entities=len(maps['id2e']),
                          preferred_device=preferred_device)

    model.fit(data['train'], epochs=epochs)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    run = mlflow.start_run(run_name=mlflow_run_name)
    run_id = run.info.run_id

    mlflow.pytorch.log_model(model, 'models/')
    mlflow.end_run()

    metrics = model.evaluate(data['test'])

    mlflow.start_run(run_id=run_id, run_name=mlflow_run_name)

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.end_run()

if __name__ == "__main__":
    main()
