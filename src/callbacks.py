from typing import Dict, Any
import torch
import logging


class Callback:
    def __init__(self, name: str = None):
        self.name = name

    def on_epoch_start(self, epoch, params: Dict[str, Any]):
        pass

    def on_epoch_end(self, epoch, params: Dict[str, Any]):
        pass

    def on_batch_start(self, params: Dict[str, Any]):
        pass

    def on_batch_end(self, params: Dict[str, Any]):
        pass


class EarlyStopping(Callback):
    def __init__(self, monitor_value="val_loss",
                 threshold=1e-4, wait_epochs=20):
        super(EarlyStopping, self).__init__('early_stopping')
        self.monitor_value = monitor_value
        self.threshold = threshold
        self.wait_epochs = wait_epochs

    def on_epoch_end(self, epoch, params: Dict[str, Any]):
        if self.monitor_value == "val_loss":
            losses = torch.Tensor(params.get(self.monitor_value))
            model = params['model']

            if losses.shape[0] < self.wait_epochs + 1:
                return

            ref_epoch = losses[-self.wait_epochs - 1]
            last_n_epochs = losses[-self.wait_epochs:] + self.threshold

            if all(ref_epoch < last_n_epochs):
                logging.info("Validation loss hasn't improved for past %d. Stopping Training now", self.wait_epochs) # noqa
                model.stop_training = True

        elif self.monitor_value in params['val_metrics'].keys():
            metrics = params['val_metrics'][self.monitor_value]
            model = params['model']

            metrics = torch.Tensor(metrics)

            if len(metrics) < self.wait_epochs + 1:
                return

            ref_epoch = metrics[-self.wait_epochs - 1]
            last_n_epochs = metrics[-self.wait_epochs:] - self.threshold

            if all(ref_epoch > last_n_epochs):
                logging.info("Validation metric %d hasn't improved for past %d. Stopping Training now", self.monitor_value, self.wait_epochs) # noqa
                model.stop_training = True
