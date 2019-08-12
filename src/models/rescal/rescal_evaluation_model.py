import torch


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class RESCAL:
    def __init__(self, embedding_dim=100, batch_size=4):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

    def get_score(self, head: torch.tensor, relation: torch.tensor,
                  tail: torch.tensor) -> torch.tensor:
        """
        Computes Scores for head, relation, tail triples with DistMult model

        :param head: torch.tensor, dtype: int, shape: (batch_size,
        sample_size, entity_dim)
        :param relation: torch.tensor, dtype: int, shape: (batch_size,
        sample_size, relation_dim)
        :param tail: torch.tensor, dtype: int, shape: (batch_size, sample_size,
        entity_dim)
        :param mode: str ('single', 'head-batch' or 'head-tail')

        :return: torch.tensor, dtype: float, shape: (batch_size, num_entities)
        """
        num_inputs = list(relation.size())[0]
        m = relation.view(-1, self.embedding_dim, self.embedding_dim)
        tail = tail.permute([0, 2, 1])
        h_m_embs = torch.matmul(head, m)
        scores = torch.matmul(h_m_embs, tail)
        scores = scores.view(num_inputs, -1)
        return scores
