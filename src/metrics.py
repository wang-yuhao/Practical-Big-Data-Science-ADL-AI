import numpy as np
import logging

from typing import Any, Dict
from abc import abstractmethod

logger = logging.getLogger(__name__)


class Metric(object):
    def __init__(self, name):
        """The base class to calculate metrics during training and evaluation.

        Parameters
        ----------
        name - str:
            A string identifying this particular metric
        """
        self.name = name

    def __call__(self, parameters: Dict[str, Any]) -> Any:
        """Calculates the metric"""
        return self.calculate_metric(parameters)

    @abstractmethod
    def calculate_metric(self, parameters: Dict[str, Any]) -> Any:
        """Subclasses should implement this method to calculate a metric.

        Parameters
        ----------
        parameters - Dict[str, Any]:
            subj_ranks: These are the individual ranks for each ranked subject
                        entity
            obj_ranks: These are the individual ranks for each rankd object
                       entity
            num_triples: The total number of positive triples that have been
                         ranked
        """

    def log_metric(self, metric):
        logger.debug('%s - %d', self.name, metric)


class HitsAtK(Metric):
    def __init__(self, k):
        super(HitsAtK, self).__init__(f'hits_at_{k}')
        self.k = k

    def calculate_metric(self, parameters: Dict[str, Any]) -> Dict[str, int]:
        """Calculates the percentage of ranks lower than or equal to k.
        """
        ranks = parameters['ranks']
        num_triples = len(ranks)

        if 'subj_ranks' in parameters or 'obj_ranks' in parameters:
            subj_ranks = parameters['subj_ranks']
            obj_ranks = parameters['obj_ranks']

            hit_s = np.sum(np.array(subj_ranks) <= self.k)
            hit_o = np.sum(np.array(obj_ranks) <= self.k)
            hits = (hit_s + hit_o) / num_triples
        else:
            hits = np.sum(np.array(ranks) <= self.k) / num_triples

        self.log_metric(hits)

        return hits.item()


class MeanReciprocalRank(Metric):
    def __init__(self):
        super(MeanReciprocalRank, self).__init__('mean_reciprocal_rank')

    def calculate_metric(self, parameters: Dict[str, Any]) -> float:
        """MRR is the average inverse rank for all test triples"""
        ranks = np.array(parameters['ranks'], dtype=float)
        mrr = np.mean(np.reciprocal(ranks))
        self.log_metric(mrr)
        return mrr


class MeanRank(Metric):
    def __init__(self):
        super(MeanRank, self).__init__('mean_rank')

    def calculate_metric(self, parameters: Dict[str, Any]) -> float:
        ranks = np.array(parameters['ranks'], dtype=float)
        mr = np.mean(ranks)
        self.log_metric(mr)
        return mr
