from src.callbacks import EarlyStopping
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def model():
    return MagicMock(stop_training=False)


@pytest.fixture
def params(model):
    return dict(
        model=model,
    )


def test_early_stopping_current_epoch_smaller(model, params):
    params['val_loss'] = torch.arange(end=1, dtype=torch.float)

    c = EarlyStopping('val_loss', wait_epochs=2)
    c.on_epoch_end(0, params)

    assert c.monitor_value == 'val_loss'
    assert c.wait_epochs > len(params['val_loss'])
    assert not model.stop_training


def test_early_stopping_current_epoch_equal(model, params):
    params['val_loss'] = torch.arange(end=2, dtype=torch.float)

    c = EarlyStopping('val_loss')
    c.on_epoch_end(0, params)

    assert c.monitor_value == 'val_loss'
    assert not model.stop_training


def test_early_stopping_decreasing(model, params):
    params['val_loss'] = torch.Tensor([3, 2, 1])

    c = EarlyStopping('val_loss')
    c.on_epoch_end(0, params)

    assert c.monitor_value == 'val_loss'
    assert not model.stop_training


def test_early_stopping_constant(model, params):
    params['val_loss'] = torch.Tensor([1, 1, 1])

    c = EarlyStopping('val_loss', wait_epochs=2)
    c.on_epoch_end(0, params)

    assert c.monitor_value == 'val_loss'
    assert model.stop_training


def test_early_stopping_flatten_out(model, params):
    params['val_loss'] = torch.Tensor([1, 1-(1e-2), 1-(1e-3)])

    c = EarlyStopping('val_loss', wait_epochs=2)
    c.on_epoch_end(0, params)

    assert c.monitor_value == 'val_loss'
    assert not model.stop_training
