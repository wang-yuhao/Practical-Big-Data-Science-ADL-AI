"""get_score for DistMult"""
import torch


class Namespace:
    """Name space init"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class DistMult:
    """Class for DistMult with score function"""
    def get_score(self, head: torch.tensor, relation: torch.tensor,
                  tail: torch.tensor, mode: str) -> torch.tensor:
        """
        Computes Scores for head, relation, tail triples with DistMult model

        :param head: torch.tensor, dtype: int, shape: (batch_size, sample_size,
        entity_dim)
        :param relation: torch.tensor, dtype: int, shape: (batch_size,
        sample_size, relation_dim)
        :param tail: torch.tensor, dtype: int, shape: (batch_size, sample_size,
        entity_dim)
        :param mode: str ('single', 'head-batch' or 'head-tail')

        :return: torch.tensor, dtype: float, shape: (batch_size, num_entities)
        """

        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score
