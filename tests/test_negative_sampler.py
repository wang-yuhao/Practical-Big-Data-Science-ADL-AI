from src.models.api import NegSampleGenerator

import numpy as np
import torch
import pytest

np.random.seed(0)
NUM_ENTITIES = 5


@pytest.fixture
def heads(num_entities=NUM_ENTITIES):
    return np.random.randint(num_entities, size=(10, 1))


@pytest.fixture
def predicates(num_entities=NUM_ENTITIES):
    return np.random.randint(num_entities, size=(10, 1))


@pytest.fixture
def tails(num_entities=NUM_ENTITIES):
    return np.random.randint(num_entities, size=(10, 1))


@pytest.fixture
def triples(heads, predicates, tails):
    return np.concatenate([heads, predicates, tails], axis=-1)


@pytest.fixture
def sampler(triples):
    yield NegSampleGenerator(triples)


def test_generate_negative_samples(
        heads, predicates, tails, sampler):
    num_entities = NUM_ENTITIES

    head = torch.tensor(heads[0], dtype=torch.int32)
    predicate = torch.tensor(predicates[0], dtype=torch.int32)
    tail = torch.tensor(tails[0], dtype=torch.int32)

    params = dict(
        head=head,
        relation=predicate,
        tail=tail,
        num_entities=num_entities
    )

    neg_samples, _ = sampler.get_neg_sample(mode='head-batch', **params)

    assert neg_samples.shape[0] == 1
    assert neg_samples.shape[1] == num_entities

    neg_samples, _ = sampler.get_neg_sample(mode='tail-batch', **params)

    assert neg_samples.shape[0] == 1
    assert neg_samples.shape[1] == num_entities


def test_generate_negative_samples_with_filter_bias():
    head = torch.tensor([[1]], dtype=torch.int32)
    predicate = torch.tensor([[0]], dtype=torch.int32)
    tail = torch.tensor([[1]], dtype=torch.int32)

    triples = np.array([[1, 0, 1],
                        [1, 0, 2],
                        [0, 1, 0]])

    nsampler = NegSampleGenerator(triples, create_filter_bias=True)

    params = dict(
        head=head,
        relation=predicate,
        tail=tail,
        num_entities=3,
        mode='tail-batch',
    )

    nsampler.create_filter_bias = True
    neg_samples, filter_bias = nsampler.get_neg_sample(**params)

    assert filter_bias is not None
    assert neg_samples.shape == (1, 3)
    assert filter_bias.sum().item() == -1

    triples = np.array([[1, 0, 0],
                        [1, 0, 1],
                        [1, 0, 2],
                        [0, 1, 0]])

    nsampler = NegSampleGenerator(triples, create_filter_bias=True)
    neg_samples, filter_bias = nsampler.get_neg_sample(**params)

    assert filter_bias is not None
    assert neg_samples.shape == (1, 3)
    assert filter_bias.sum().item() == -2
