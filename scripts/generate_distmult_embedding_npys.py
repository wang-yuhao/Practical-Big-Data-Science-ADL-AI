import torch
import os
from os.path import join
import numpy as np


def main():
    path = 'models/20190630-080153-DistMult.pickle'
    checkpoint = torch.load(path)

    save_path = join(os.pardir, os.pardir, os.pardir, 'models/distmult/')

    entity_path = join(save_path, 'entity_embedding.npy')
    relation_path = join(save_path, 'relation_embedding.npy')
    ent_emb = checkpoint['entity_embeddings.weight'].cpu().data.numpy()
    np.save(entity_path, ent_emb)
    rel_emb = checkpoint['relation_embeddings.weight'].cpu().data.numpy()
    np.save(relation_path, rel_emb)


if __name__ == '__main__':
    main()
