"""This is for extracting embeddings from with pykeen trained models"""
import os
import numpy as np
import torch

def main():
    """Main for script"""
    path = 'models/20190630-080153-DistMult.pickle'
    checkpoint = torch.load(path)

    save_path = os.path.join(os.pardir, os.pardir, os.pardir, 'models/distmult/')

    entity_path = os.path.join(save_path, 'entity_embedding.npy')
    relation_path = os.path.join(save_path, 'relation_embedding.npy')
    ent_emb = checkpoint['entity_embeddings.weight'].cpu().data.numpy()
    np.save(entity_path, ent_emb)
    rel_emb = checkpoint['relation_embeddings.weight'].cpu().data.numpy()
    np.save(relation_path, rel_emb)


if __name__ == '__main__':
    main()
