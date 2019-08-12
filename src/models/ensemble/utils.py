import numpy
import logging
import tqdm

from collections import defaultdict
from typing import Tuple


def to_multihot(
        positive_triples: numpy.ndarray,
        ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Convert an array of positive triples into the multi-hot format.

    This is done by grouping every triple (s, p, o) having the same (s, p)
    into a single row, and constructing a set of related objects, e.g.

    s, p, o1
    s, p, o2
    s, p, o5

    are converted to

    s, p, {o1, o2, o5}


    :param positive_triples: numpy.ndarray, shape: (n, 3), dtype: int
        The positive triples in (s, p, o) format.
    :return: A tuple (x, y) where
        x: numpy.ndarray, shape: (m, 2), dtype: int (m <= n)
            The unique (s, p) combinations.
        y: numpy.ndarray, shape: (m,), dtype: object
            An array of lists containing all objects related to a (s, p)-pair.
    """

    logging.info('Start converting %d triples to multi-hot.', len(positive_triples))  # noqa

    # Collect (s, p, ?)
    triple_dict = defaultdict(set)
    for row in tqdm.tqdm(positive_triples, unit='triple', unit_scale=True):
        triple_dict[(row[0], row[1])].add(row[2])

    # Create lists out of sets for proper numpy indexing when loading
    # the labels
    triple_dict = {key: sorted(value) for key, value in triple_dict.items()}

    # Convert to numpy arrays
    x, y = [numpy.array(a) for a in zip(*triple_dict.items())]

    logging.info('Converted %d triples to %d multi-hot samples.',len(positive_triples), len(x))  # noqa

    return x, y
