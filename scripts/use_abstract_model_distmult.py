# coding=utf-8
import torch
from torch import nn
import numpy as np

from src.models.api import AbstractModel, evaluate


class DistMult(AbstractModel):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
    ):
        super(DistMult, self).__init__(
            num_entities=num_entities,
            num_relations=num_relations
        )
        self.entity_embedding = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim
        )
        self.relation_embedding = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim
        )

    def score_subjects(
            self,
            p: torch.tensor,
            o: torch.tensor,
    ) -> torch.tensor:
        p_emb = self.relation_embedding(p)
        o_emb = self.entity_embedding(o)
        all_emb = self.entity_embedding.weight.data
        return torch.sum(
            all_emb * p_emb[None, :] * o_emb[None, :],
            dim=-1
        )

    def score_objects(
            self,
            s: torch.tensor,
            p: torch.tensor,
    ) -> torch.tensor:
        s_emb = self.entity_embedding(s)
        p_emb = self.relation_embedding(p)
        all_emb = self.entity_embedding.weight.data
        return torch.sum(
            s_emb[None, :] * p_emb[None, :] * all_emb,
            dim=-1
        )

    def forward(self, *inputs):
        raise Exception("Not implemented")


if __name__ == '__main__':
    model = DistMult(num_entities=128, num_relations=16, embedding_dim=64)
    n_triples = 256
    device = torch.device('cpu')
    sbjs = np.random.randint(
        model.num_entities,
        size=(n_triples,),
        dtype=np.int32
    )
    pred = np.random.randint(
        model.num_relation,
        size=(n_triples,),
        dtype=np.int32
    )
    objs = np.random.randint(
        model.num_entities,
        size=(n_triples,),
        dtype=np.int32
    )
    fake_triples = np.stack([sbjs, pred, objs], axis=-1)

    results = evaluate(triples=fake_triples, model=model, device=device)
    print(results)
