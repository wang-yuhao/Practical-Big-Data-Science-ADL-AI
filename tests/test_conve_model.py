import torch
import pytest

import numpy as np
from unittest.mock import patch

from src.models.conve import ConvE
from src.models.api import evaluate


@pytest.fixture
def model():
    model = ConvE(
                num_entities=10,
                num_relations=10,
                embedding_dim=200,
                ConvE_output_channels=3,
                ConvE_input_channels=1,
                ConvE_height=10,
                ConvE_width=20,
                ConvE_kernel_height=3,
                ConvE_kernel_width=3,
                conv_e_feature_map_dropout=0.1,
                conv_e_input_dropout=0.1,
                conv_e_output_dropout=0.3,
                preferred_device='cpu'
            )
    model.compile()
    yield model


@pytest.fixture
def fake_triples(model):
    n_triples = 256

    s = np.random.randint(
        model.num_entities,
        size=(n_triples,),
        dtype=np.long
    )

    p = np.random.randint(
        model.num_entities,
        size=(n_triples,),
        dtype=np.long
    )

    o = np.random.randint(
        model.num_entities,
        size=(n_triples,),
        dtype=np.long
    )

    return np.stack([s, p, o], axis=-1)


def test_expand_relation_inverse_model(model):
    """
    Shouldn't modify the original relation when evaluated on the inverse model.

    The left-side-prediction should be called with reversed relations.
    In the following all config parameters for the ConvE model are
    ignored, except the number of relations.
    When trained with the inverse model, we create new 'reversed'
    relations by increasing the original realtion id by num_relations.
    """
    with patch.object(ConvE, 'predict_for_ranking', return_value=None) as spy:
        model.inverse_model = True
        model.num_relations *= 2
        model._init_embeddings()

        o = torch.tensor([1], dtype=torch.long)
        r = torch.tensor([2], dtype=torch.long)
        _ = model.score_subjects(p=r, o=o)

    spy.assert_called_once_with(o, r + model.num_relations // 2)
    assert r.item() == 2


def test_global_evaluate_should_not_fail(model, fake_triples):
    device = torch.device('cpu')
    model._init_embeddings()

    results = evaluate(triples=fake_triples, model=model, device=device)

    print(results)


def test_model_evaluation_should_not_fail(model, fake_triples):
    model._init_embeddings()

    fake_train = fake_triples.copy()
    fake_test = fake_triples.copy()

    results = model.evaluate(train_triples=fake_train,
                             test_triples=fake_test)

    print(results)
