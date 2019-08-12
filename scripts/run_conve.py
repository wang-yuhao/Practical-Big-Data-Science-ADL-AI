# %% [markdown]
# # ConvE
# This notebook is used to train and evaluate the ConvE model. Since
# the adlai library hasn't been installed yet we need to add the path
# to the source code, so we can import the ConvE model.

# %% Define all dependencies

import sys
import os
import mlflow.pytorch
import mlflow
import logging
import click
import numpy as np
import torch

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('src/models/'))

from src.models.conve import ConvE  # noqa
from src.data.utils import load # noqa
from src.metrics import HitsAtK, MeanRank, MeanReciprocalRank # noqa
from src.callbacks import Callback # noqa

logging.basicConfig(level=logging.DEBUG)

log_dir = f'/tmp/run-conve-logs/'
os.path.isdir(log_dir)
os.makedirs(log_dir, exist_ok=True)

print(f'{"*" * 5} Logging to {log_dir} {"*" * 5}')

fmt="%(asctime)s %(levelname)-8s %(name)-15s %(message)s" # noqa

fileHandler = logging.FileHandler(os.path.join(log_dir, '{time.time()}-conve.log'))

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.addHandler(handler)
logger.addHandler(fileHandler)

# General Parameters
DEVICE_PARAM = 'cpu'

TRAIN_REVERSED_PARAM = True
LABEL_SMOOTHING_PARAM = 0.1
WEIGHT_DECAY_PARAM = 0.0

VALIDATION_INTERVAL_PARAM = 10
MLFLOW_EXPERIMENT_NAME_PARAM = 'ConvE'

# Default Hyperparameters
EPOCHS_PARAM = 1000
BATCH_SIZE_PARAM = 128
LR_PARAM = 1e-3

# Model Parameters
CONVE_INPUT_CHANNELS = 1
CONVE_OUTPUT_CHANNELS = 2

CONVE_EMBEDDING_DIM = 200
CONVE_EMBEDDING_WIDTH = 20
CONVE_EMBEDDING_HEIGHT = 10

CONVE_KERNEL_WIDTH = 3
CONVE_KERNEL_HEIGHT = 3

CONVE_EMBEDDING_DROPOUT = 0.2
CONVE_FEATURE_MAP_DROPOUT = 0.2
CONVE_PROJECTION_DROPOUT = 0.3

# Help
HELP_DEVICE = f'Either cuda or cpu. (default: {DEVICE_PARAM})'

HELP_TRAIN_REVERSED = f'If True, creates reversed triples so left-prediction\
                       can be transformed into a right-prediction.\
                       (default: {TRAIN_REVERSED_PARAM})'

HELP_EPOCHS = f'Sets the number of epochs to train.\
               (default: {EPOCHS_PARAM})'

HELP_BATCH_SIZE = f'Sets the batch size.\
                   (default: {BATCH_SIZE_PARAM})'

HELP_LR = f'Sets the learning rate:\
           (default: {LR_PARAM})'

HELP_LABEL_SMOOTHING = f"If set, applies label smoothing to the training labels.\
                        (default: {LABEL_SMOOTHING_PARAM})"

HELP_WEIGHT_DECAY = f"Sets the weight decay parameter for the Adam optimzer.\
                     (default: {WEIGHT_DECAY_PARAM})"

HELP_VALIDATION_INTERVAL = f"The interval at which the model is validated\
                            on the validation set.\
                            (default: {VALIDATION_INTERVAL_PARAM})"

HELP_MLFLOW_EXPERIMENT_NAME = f'If set, changes the name of the experiment\
                               under which model artifacts are saved.\
                               (default: {MLFLOW_EXPERIMENT_NAME_PARAM})'

def get_model(num_entities, num_relations, device):
    '''Initializes a ConvE model.
    Parameters
    ----------
        num_entities - int: The total number of distinct entities in the
                            dataset.
        num_relations - int: The total number of distinct realtions in the
                             dataset.
    '''
    model = ConvE(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=CONVE_EMBEDDING_DIM,
        ConvE_input_channels=CONVE_INPUT_CHANNELS,
        ConvE_output_channels=CONVE_OUTPUT_CHANNELS,
        ConvE_width=CONVE_EMBEDDING_WIDTH,
        ConvE_height=CONVE_EMBEDDING_HEIGHT,
        ConvE_kernel_height=CONVE_KERNEL_HEIGHT,
        ConvE_kernel_width=CONVE_KERNEL_WIDTH,
        conv_e_input_dropout=CONVE_EMBEDDING_DROPOUT,
        conv_e_feature_map_dropout=CONVE_FEATURE_MAP_DROPOUT,
        conv_e_output_dropout=CONVE_PROJECTION_DROPOUT,
        preferred_device=device
    )
    return model


@click.command()
@click.option('--epochs',type=int, default=EPOCHS_PARAM, help=HELP_EPOCHS)
@click.option('--batch_size',type=int, default=BATCH_SIZE_PARAM, help=HELP_BATCH_SIZE)
@click.option('--lr', type=float, default=LR_PARAM, help=HELP_LR)
@click.option('--device', default=DEVICE_PARAM, help=HELP_DEVICE)
@click.option('--train_reversed', default=TRAIN_REVERSED_PARAM, type=bool, help=HELP_TRAIN_REVERSED) # noqa
@click.option('--label_smoothing', default=LABEL_SMOOTHING_PARAM, type=float, help=HELP_LABEL_SMOOTHING) # noqa
@click.option('--weight_decay', default=0, type=float, help=HELP_WEIGHT_DECAY)
@click.option('--validation_interval', default=VALIDATION_INTERVAL_PARAM, help=HELP_VALIDATION_INTERVAL)
@click.option('--mlflow_experiment_name', default=MLFLOW_EXPERIMENT_NAME_PARAM, help=HELP_MLFLOW_EXPERIMENT_NAME) # noqa
@click.option('--embedding_dim', type=int, default=CONVE_EMBEDDING_DIM, help=f'(default: {CONVE_EMBEDDING_DIM})')
@click.option('--embedding_height', type=int, default=CONVE_EMBEDDING_HEIGHT, help=f'(default: {CONVE_EMBEDDING_HEIGHT})')
@click.option('--embedding_width', type=int, default=CONVE_EMBEDDING_WIDTH, help=f'(default: {CONVE_EMBEDDING_WIDTH})')
@click.option('--input_channels', type=int, default=CONVE_INPUT_CHANNELS, help=f'(default: {CONVE_INPUT_CHANNELS})')
@click.option('--output_channels', type=int, default=CONVE_OUTPUT_CHANNELS, help=f'(default: {CONVE_OUTPUT_CHANNELS})')
@click.option('--kernel_height', type=int, default=CONVE_KERNEL_HEIGHT, help=f'(default: {CONVE_KERNEL_HEIGHT})')
@click.option('--kernel_width', type=int, default=CONVE_KERNEL_WIDTH, help=f'(default: {CONVE_KERNEL_WIDTH})')
@click.option('--embedding_dropout', type=float, default=CONVE_EMBEDDING_DROPOUT, help=f'(default: {CONVE_EMBEDDING_DROPOUT})')
@click.option('--feature_map_dropout', type=float, default=CONVE_FEATURE_MAP_DROPOUT, help=f'(default: {CONVE_FEATURE_MAP_DROPOUT})')
@click.option('--projection_dropout', type=float, default=CONVE_PROJECTION_DROPOUT, help=f'(default: {CONVE_PROJECTION_DROPOUT})')
# %% Train the model
def main(epochs, batch_size, lr, device, train_reversed,
         label_smoothing, weight_decay,  validation_interval, mlflow_experiment_name,
         embedding_dim,embedding_height, embedding_width, input_channels, output_channels,
         kernel_height, kernel_width, embedding_dropout, feature_map_dropout,
         projection_dropout
         ):
    """
    This script trains a ConvE KGE model and validates the model every 10 epochs
    (--validation_interval).
    If the MRR score has improved since the last checkpoint a new checkpoint
    will be logged to the mlflow server.
    """
    mappings, datasets = load()
    train = datasets['train']
    valid = datasets['valid']
    test = datasets['test']

    id2rel = mappings['id2rel']
    id2e = mappings['id2e']

    num_entities = len(id2e)
    num_relations = len(id2rel)

    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.start_run()

    mlflow.log_param('Epochs', epochs)
    mlflow.log_param('Label Smoothing', label_smoothing)
    mlflow.log_param('train_reversed', train_reversed)
    mlflow.log_param('Batch Size', batch_size)
    mlflow.log_param('Learning Rate', lr)
    mlflow.log_param('Weight Decay', weight_decay)

    mlflow.log_param('num_entities', num_entities)
    mlflow.log_param('num_relations', num_relations)
    mlflow.log_param('embedding_dim', embedding_dim)
    mlflow.log_param('ConvE_input_channels', input_channels)
    mlflow.log_param('ConvE_output_channels', output_channels)
    mlflow.log_param('ConvE_width', embedding_width)
    mlflow.log_param('ConvE_height', embedding_height)
    mlflow.log_param('ConvE_kernel_height', kernel_height)
    mlflow.log_param('ConvE_kernel_width', kernel_width)
    mlflow.log_param('ConvE_input_dropout', embedding_dropout)
    mlflow.log_param('ConvE_feature_map_dropout', feature_map_dropout)
    mlflow.log_param('ConvE_output_dropout', projection_dropout)

    model = get_model(num_entities, num_relations, device)
    model.compile(metrics=[HitsAtK(1),
                           HitsAtK(3),
                           HitsAtK(10),
                           MeanRank(),
                           MeanReciprocalRank()],
                  callbacks=[
                     MlFlowLogger(mlflow),
                     SaveModel('mean_reciprocal_rank', objective='max')
                     ])

    try:
        losses, _ = model.fit(train,
                              valid,
                              learning_rate=lr,
                              num_epochs=epochs,
                              train_reversed=train_reversed,
                              label_smoothing=label_smoothing,
                              weight_decay=weight_decay,
                              validation_interval=validation_interval,
                              batch_size=batch_size)
    except KeyboardInterrupt:
        logger.warning('Forced Keyboard Interrupt. Exiting now...')
        mlflow.log_param('training_interrupted', True)
        sys.exit()

    epochs = len(losses)

    mlflow.log_param('Early Stop Epochs', epochs)

    for epoch in range(epochs):
        mlflow.log_metric('loss', losses[epoch], step=epoch)

    logger.info("*"*30)
    logger.info("Evaluating on Test set")
    logger.info("*"*30)

    with torch.no_grad():
        results = model.evaluate(test, train)

    for epoch, item in enumerate(results.items()):
        mlflow.log_metric(item[0], item[1], step=epoch)

    mlflow.pytorch.log_model(model, 'models/conve-model-final')

    mlflow.end_run()

    model.eval()

    h = torch.tensor(test[0, 0:1], dtype=torch.long, device=device)
    p = torch.tensor(test[0, 1:2], dtype=torch.long, device=device)
    t = torch.tensor(test[0, 1:2], dtype=torch.long, device=device)

    obj_scores = model.score_objects(h, p).detach()
    subj_scores = model.score_subjects(p, t).detach()

    print('Object Scores')
    print(f'(min/max): ({obj_scores.min()}, {obj_scores.max()})')
    o_sort, o_args = torch.sort(obj_scores, descending=True)
    print('Sorted Top 10 Scores: ', o_sort[:10])
    print('Sorted Top 10 Ids: ', o_args[:10])
    print('True Id: ', t.item())

    print('Subject Scores')
    print(f'(min/max): ({subj_scores.min()}, {subj_scores.max()})')
    s_sort, s_args = torch.sort(subj_scores, descending=True)
    print('Sorted Top 10 Scores: ', s_sort[:10])
    print('Sorted Top 10 Ids: ', s_args[:10])
    print('True Id: ', h.item())


class MlFlowLogger(Callback):
    def __init__(self, mlflow_intance):
        super(MlFlowLogger, self).__init__('MlFlowLogger')
        self.mlflow_intance = mlflow_intance
        self.prev_epoch = None

    def on_epoch_end(self, epoch, params):
        metrics = params['val_metrics']

        if self.prev_epoch != epoch:
            self.prev_epoch = epoch
            for k, v in metrics.items():
                self.mlflow_intance.log_metric(k, v[-1], step=epoch)


class SaveModel(Callback):
    def __init__(self, monitor_value, objective=None):
        super(SaveModel, self).__init__('SaveModel')

        self.monitor_value = monitor_value
        self.current_best = None

        if objective == 'min':
            self.objective = np.less
        elif objective == 'max':
            self.objective = np.greater

    def on_epoch_end(self, epoch, params):
        if self.monitor_value not in params['val_metrics']:
            raise Exception(f'Metric: {self.monitor_value} not found. Please include the metric during model compilation') # noqa

        model = params['model']
        metric = params['val_metrics']
        most_recent_metric = metric[self.monitor_value][-1]

        file_name = f'conve-model-{epoch}'

        if not self.current_best \
           or self.objective(most_recent_metric, self.current_best):

            logger.debug('New Best Model: [%d] %d --> %d', self.monitor_value, self.current_best, most_recent_metric) # noqa

            for k, v in params['val_metrics'].items():
                logger.debug('%s: %d', k, v[-1])

            self.current_best = most_recent_metric
            mlflow.pytorch.log_model(model, f'models/{file_name}')

#pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
