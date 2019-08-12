import os
import torch
import click
import logging
import tempfile
import numpy as np

from src.models.conve import ConveEvaluationModel
from src.data.utils import load

import mlflow
import mlflow.pytorch

logging.getLogger('src.model.api').setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



PARAM_EXPERIMENT = 'conve-error-analysis'
HELP_EXPERIMENT = f"If provided, creates a new experiment on the mlflow\
                    server with the given name.\
                    (default: {PARAM_EXPERIMENT})"

HELP_ARTIFACT_PATH = 'The artifact path to trained model chkpt file.'

PARAM_DEVICE = 'cpu'
HELP_DEVICE = f'Either cuda or cpu. (default: {PARAM_DEVICE})'

PARAM_RUN_NAME = None
HELP_RUN_NAME = f'A tag to identify the evaluation output on the mlflow server.\
                  (default: {PARAM_RUN_NAME})'

PARAM_TRACKING_URI = 'http://10.195.1.54'
HELP_TRACKING_URI = f'Sets the tracking uri for the mflow instance.\
                      Set this option to `file:/my/local/dir` to store\
                      artifacts locally.\
                      (default: {PARAM_TRACKING_URI})'


def load_remote_model(device, artifact_path):
    if device == 'cpu':
        model = mlflow.pytorch.load_model(artifact_path, map_location='cpu')
    else:
        model = mlflow.pytorch.load_model(artifact_path)

    model.to(device=device)
    return model


@click.command()
@click.option('--artifact_path', help=HELP_ARTIFACT_PATH)
@click.option('--device', default=PARAM_DEVICE, help=HELP_DEVICE)
@click.option('--run_name', default=PARAM_RUN_NAME, help=HELP_RUN_NAME)
@click.option('--mlflow_experiment_name', default=PARAM_EXPERIMENT, help=HELP_EXPERIMENT)
@click.option('--mlflow_tracking_uri', default=PARAM_TRACKING_URI, help=HELP_TRACKING_URI)
def run_evaluation_cmd(artifact_path=None, device=None, run_name=None,
                       mlflow_experiment_name=None, mlflow_tracking_uri=None):
    """
    This script evaluates a trained conve model and logs the
    results to an mlflow tracking server.
    """
    run_evaluation(artifact_path, device, run_name,
                   mlflow_experiment_name, mlflow_tracking_uri)


def run_evaluation(artifact_path=None,
                   device='cpu',
                   run_name=None,
                   eval_experiment_name='conve-error-analysis',
                   tracking_uri=None):
    logger.info('*' * 30)
    logger.info('Evaluate on: %s', device) 
    logger.info('*' * 30)

    model = load_remote_model(device, artifact_path)

    maps, data = load()

    num_entities = len(maps['id2e'])

    train, valid, test = data['train'], data['valid'], data['test']

    all_positive_triples = np.concatenate([train, valid, test], axis=0)
    all_positive_triples = torch.tensor(all_positive_triples,
                                        dtype=torch.long,
                                        device=device)

    all_entities = torch.arange(num_entities,
                                dtype=torch.long,
                                device=device)

    test = torch.tensor(test,
                        dtype=torch.long,
                        device=device)

    evalModel = ConveEvaluationModel(model,
                                     all_positive_triples,
                                     all_entities,
                                     device=device)

    results = evalModel.evaluate(
        s=test[:, 0],
        p=test[:, 1],
        o=test[:, 2],
        batch_size=128
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(eval_experiment_name)
        mlflow.start_run(run_name=run_name)
        evalModel.generate_model_output(output_path=tmpdir,
                                        test_triples=test,
                                        evaluation_result=results)

        files = os.listdir(tmpdir)

        for f in files:
            mlflow.log_artifact(os.path.join(tmpdir, f))

        mlflow.end_run()

if __name__ == '__main__':
    run_evaluation_cmd()
