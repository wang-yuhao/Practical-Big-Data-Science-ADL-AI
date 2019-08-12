import torch
import abc
from torch import nn

from pykeen.kge_models import RESCAL, TransE, DistMult


class AbstractModel(nn.Module):
    
    #pylint: disable=arguments-differ
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Compute the scores for given s, p and every o.

        :param x: torch.tensor, dtype: torch.long, shape: (batch_size, 2)
            The s, p indices.

        :return:
            scores: torch.tensor, dtype: torch.float, shape: (batch_size, num_entities) # noqa
                The scores for every possible entity as o.
        """
        raise NotImplementedError()


class EnsembleUnit(abc.ABC, AbstractModel):
    def __init__(self, model=None):
        super(EnsembleUnit, self).__init__()
        self.model = model

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Compute the scores for given s, p and every o.

        :param x: torch.tensor, dtype: torch.long, shape: (batch_size, 2)
            The s, p indices.

        :return:
            scores: torch.tensor, dtype: torch.float, shape: (batch_size, num_entities) # noqa
                The scores for every possible entity as o.
        """
        raise NotImplementedError()


class RescalUnit(EnsembleUnit):
    def __init__(self, num_entities, num_relations, relation_embeddings,
                 entity_embeddings, preferred_device='cpu'):
        super(RescalUnit, self).__init__()

        if preferred_device == 'cuda':
            preferred_device = 'gpu'

        self.model = RESCAL(
            preferred_device=preferred_device,
            random_seed=0,
            embedding_dim=entity_embeddings.shape[1],
            margin_loss=1,
            scoring_function=2,
        )

        self.model.num_entities = num_entities
        self.model.num_relations = num_relations
        self.model.relation_embeddings = nn.Embedding.from_pretrained(relation_embeddings) # noqa
        self.model.entity_embeddings = nn.Embedding.from_pretrained(entity_embeddings) # noqa

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Compute the scores for given s, p and every o.

        :param x: torch.tensor, dtype: torch.long, shape: (batch_size, 2)
            The s, p indices.

        :return:
            scores: torch.tensor, dtype: torch.float, shape: (batch_size, num_entities) # noqa
                The scores for every possible entity as o.
        """
        # batch_size = x.shape[0]

        # num_inputs = list(relation.size())[0]
        # m = relation.view(-1,
        #                   self.model.embedding_dim,
        #                   self.model.embedding_dim)
        # h_m_embs = torch.matmul(head, m)
        # tail_embeddings = self.model.entity_embeddings.weight.view(1,
        #                                                            num_entities,
        #                                                            entity_embedding_size)
        # tail_embeddings = tail_embeddings.permute([0, 2, 1])
        # scores = torch.matmul(h_m_embs, tail)
        # scores = scores.view(num_inputs, -1)

        # return scores
        raise Exception('WIP')


class DistMultUnit(EnsembleUnit):
    def __init__(self, relation_embeddings, num_entities,
                 num_relations, entity_embeddings, preferred_device='cpu'):
        super(DistMultUnit, self).__init__()

        if preferred_device == 'cuda':
            preferred_device = 'gpu'

        self.model = DistMult(
            preferred_device=preferred_device,
            random_seed=1234,
            embedding_dim=entity_embeddings.shape[1],
            margin_loss=1,
            optimizer=torch.optim.Adam
        )

        self.model.num_entities = num_entities
        self.model.num_relations = num_relations
        self.model.relation_embeddings = nn.Embedding.from_pretrained(relation_embeddings) # noqa
        self.model.entity_embeddings = nn.Embedding.from_pretrained(entity_embeddings)  # noqa

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Compute the scores for given s, p and every o.

        :param x: torch.tensor, dtype: torch.long, shape: (batch_size, 2)
            The s, p indices.

        :return:
            scores: torch.tensor, dtype: torch.float, shape: (batch_size, num_entities) # noqa
                The scores for every possible entity as o.
        """
        batch_size = x.shape[0]

        head_embeddings = self.model.entity_embeddings(x[:, 0:1]).view(batch_size, 1, -1) # noqa
        relation_embeddings = self.model.relation_embeddings(x[:, 1:2]).view(batch_size, 1, -1) # noqa

        num_entities, entity_embedding_size = self.model.entity_embeddings.weight.shape # noqa
        tail_embeddings = self.model.entity_embeddings.weight.view(1, num_entities, entity_embedding_size)  # noqa

        scores = - torch.sum(head_embeddings * relation_embeddings * tail_embeddings, dim=-1) # noqa

        return scores


class TranseUnit(EnsembleUnit):
    def __init__(self, relation_embeddings, num_entities,
                 num_relations, entity_embeddings, preferred_device='cpu'):
        super(TranseUnit, self).__init__()

        if preferred_device == 'cuda':
            preferred_device = 'gpu'

        self.model = TransE(
            preferred_device=preferred_device,
            random_seed=1234,
            embedding_dim=entity_embeddings.shape[1],
            margin_loss=1,
            normalization_of_entities=1
        )

        self.model.num_entities = num_entities
        self.model.num_relations = num_relations
        self.model.relation_embeddings = nn.Embedding.from_pretrained(relation_embeddings) # noqa
        self.model.entity_embeddings = nn.Embedding.from_pretrained(entity_embeddings) # noqa

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Compute the scores for given s, p and every o.

        :param x: torch.tensor, dtype: torch.long, shape: (batch_size, 2)
            The s, p indices.

        :return:
            scores: torch.tensor, dtype: torch.float, shape: (batch_size, num_entities) # noqa
                The scores for every possible entity as o.
        """
        batch_size = x.shape[0]

        head_embeddings = self.model.entity_embeddings(x[:, 0:1])
        head_embeddings = head_embeddings.view(batch_size, 1, -1)

        relation_embeddings = self.model.relation_embeddings(x[:, 1:2])
        relation_embeddings = relation_embeddings.view(batch_size, 1, -1)

        num_entities, entity_embedding_size = self.model.entity_embeddings.weight.shape # noqa
        tail_embeddings = self.model.entity_embeddings.weight
        tail_embeddings = tail_embeddings.view(1,
                                               num_entities,
                                               entity_embedding_size)

        sum_res = head_embeddings + relation_embeddings - tail_embeddings
        distances = torch.norm(sum_res, dim=-1, p=1)

        return distances
