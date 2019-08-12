import torch


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TransE:

    def __init__(self, gamma):
        self.gamma = gamma

    def get_score(self, head: torch.tensor, relation: torch.tensor,
                  tail: torch.tensor, mode: str) -> torch.tensor:
        """
        Computes Scores for head, relation, tail triples with TransE model

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
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score