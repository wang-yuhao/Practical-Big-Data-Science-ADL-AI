import pytest
import pickle
from os import path


@pytest.fixture
def data_dir():
    file_dir = path.dirname(__file__)
    return path.join(file_dir, "../data/processed/FB15k-237/")


@pytest.fixture
def entity_to_id(data_dir):
    e2i_path = path.join(data_dir, "entity_to_id.pickle")

    with open(path.abspath(e2i_path), "rb") as handle1:
        yield pickle.load(handle1)


@pytest.fixture
def id_to_entity(data_dir):
    i2e_path = path.join(data_dir, "id_to_entity.pickle")

    with open(path.abspath(i2e_path), "rb") as handle1:
        yield pickle.load(handle1)


@pytest.fixture
def relation_to_id(data_dir):
    filename_rel_to_id = path.join(data_dir, "relation_to_id.pickle")

    with open(path.abspath(filename_rel_to_id), "rb") as handle1:
        yield pickle.load(handle1)


@pytest.fixture
def id_to_relation(data_dir):
    filename_id_to_rel = path.join(data_dir, "id_to_relation.pickle")

    with open(path.abspath(filename_id_to_rel), "rb") as handle2:
        yield pickle.load(handle2)


@pytest.fixture
def prepped_datasets(data_dir):
    ds = ["train", "valid", "test"]
    pickled = [path.join(data_dir, d + ".pickle") for d in ds]
    pickled = [open(path.abspath(p), 'rb') for p in pickled]
    yield [pickle.load(p) for p in pickled]


class TestStringMethods:

    def test_entity_dictionaries(self, entity_to_id, id_to_entity) -> None:
        """ checks whether the dictionary mappings (entities) are correct
        """
        for entity, i in entity_to_id.items():
            assert entity == id_to_entity[i]

        for i, entity in id_to_entity.items():
            assert i == entity_to_id[entity]

    def test_relation_dictionaries(self,
                                   relation_to_id, id_to_relation) -> None:
        """ checks whether the dictionary mappings (relations) are correct
        """
        for relation, i in relation_to_id.items():
            assert relation == id_to_relation[i]

        for i, relation in id_to_relation.items():
            assert i == relation_to_id[relation]

    def test_processed_datesets(self, prepped_datasets) -> None:
        """ checks whether every triple really consists of 3 elements
        """
        for triples in prepped_datasets:
            for triple in triples:
                assert len(triple) == 3

    def test_mapping(self, data_dir, id_to_entity, id_to_relation) -> None:
        """ checks the mapping between the raw and processed dataset
        """
        filenames = ["train.txt", "valid.txt", "test.txt"]

        for fname in filenames:
            filename_raw = path.join(data_dir, '../../raw/FB15k-237', fname)

            with open(path.abspath(filename_raw), 'r') as f:
                raw_triples = [l.strip().split('\t') for l in f.readlines()]

            filename_triples = path.join(data_dir, f"{fname[:-4]}.pickle")

            with open(path.abspath(filename_triples), "rb") as handle:
                processed_triples = pickle.load(handle)

            for (triple1, triple2) in zip(raw_triples, processed_triples):
                assert triple1[0] == id_to_entity[triple2[0]]
                assert triple1[1] == id_to_relation[triple2[1]]
                assert triple1[2] == id_to_entity[triple2[2]]
