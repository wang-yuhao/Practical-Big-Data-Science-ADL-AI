import numpy as np

from os.path import join


def split_list_in_batches(input_list: np.ndarray, batch_size: int):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]


# %% Load Data
def load(data_dir='data/processed/FB15k-237/'):
    '''Loads the vocabulary and the dataset.

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
    base_path = join(data_dir)

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
