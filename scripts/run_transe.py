import argparse
import logging
import torch
import time
import os

import numpy as np

from os.path import join
from src.models.transe.TransE import TransE,compute_metric_results
from mlflow import log_metric, log_param

import mlflow

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logging.getLogger('pykeen').setLevel(logging.INFO)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='run_transe.py [<args>] [-h | --help]'
    )
    parser.add_argument('-e', '--num_epochs', default=3000, type=float, help='The number of epochs')
    parser.add_argument('-lr', '--learning_rate', default=0.0005, type=float, help='The learning rate in training model')
    parser.add_argument('-b', '--batch_size', default=100, type=float, help="The batch size ")
    parser.add_argument('-p','--preferred_device', default='gpu', type=str,help="The preferred device,only 'gpu' or 'cpu'")
    parser.add_argument('-r', '--random_seed', default=0, type=float, help="The random seed")
    parser.add_argument('-dim', '--embedding_dim', default=100, type=float,help='The dimensions of embeddings')
    parser.add_argument('-g', '--margin_loss', default=12.0, type=float, help='The margin of loss function')
    parser.add_argument('-n', '--normalization_of_entities', default=1, type=float,help='The normalization of entities, only 1 or 2 ')
    parser.add_argument('-opt', '--optimizer', default='SGD', type=str, help='The optimizer for update weights, only "SGD" or "adam"')
    parser.add_argument('--filter_neg_triples', dest='filter_neg_triples', action='store_true',help='If to filter positive triples from negative triples, filter negative triples')
    parser.add_argument('--no_filter_neg_triples', dest='filter_neg_triples', action='store_false', help='If to filter positive triples from negative triples, not filter negative triples')
    parser.add_argument('-d','--data_path', default='../data/processed/FB15k-237/',type=str, help='The directory of the train, test and validate dataset')
    parser.add_argument('-o','--output', default='../models',type=str,  help='The directory to save models and trained embeddings')

    return parser.parse_args(args)



def load_datasets(data_path):
    '''Reads in the index mappings and dataset of the FB15k-237 dataset.
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
    base_path = join(data_path)

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


def get_model(args,num_entities, num_relations):
    '''Initializes a TransE model.
    Parameters
    ----------
        num_entities - int: The total number of distinct entities in the
                            dataset.
        num_relations - int: The total number of distinct realtions in the
                             dataset.
    '''
    model = TransE(
        preferred_device=args.preferred_device,
        random_seed=args.random_seed,
        embedding_dim=args.embedding_dim,
        margin_loss=args.margin_loss,
        normalization_of_entities=args.normalization_of_entities
    )

    model.num_entities = num_entities
    model.num_relations = num_relations

    # model._init_embeddings()

    return model

#@click.command()
#@click.option('--output', help='Path to output training results', default='model') # noqa
#@click.option('--data', help='Path to processes data', default='data/processed/FB15k-237/') #noa
#def main(output=None, data=None):
def main(args):
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    output = args.output

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD
    elif args.optimizer == 'adam':
        optimizer = torch.optim.adam

    preferred_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(preferred_device)

    vocab, datasets = load_datasets(args.data_path)

    train, test = datasets['train'], datasets['test']

    id2e, id2rel = vocab['id2e'], vocab['id2rel']
    all_entities = np.arange(len(id2e))

    model = get_model(args,num_entities=len(id2e), num_relations=len(id2rel))

    if os.environ.get('DEV'):
        logger.warning('Running in dev mode')

        num_epochs = 1
        train = train[:100]
        test = test[:20]

    mlflow.set_experiment("TransE")
    with mlflow.start_run():

        log_param("Epochs", num_epochs)
        log_param("Learning Rate", learning_rate)
        log_param("Batch size", batch_size)
        log_param("normalization_of_entities", model.l_p_norm_entities)

        loss_per_epoch = model.fit(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            pos_triples=train,
            tqdm_kwargs=None,
            optimizer=optimizer
        )

        # %% Save the model and the loss
        timestr = time.strftime("%Y%m%d-%H%M%S")

        torch.save(
            model.state_dict(), os.path.join(output, f'{timestr}-TransE.pickle')
        )

        torch.save(
            loss_per_epoch, os.path.join(output, f'{timestr}-TransE-loss.pickle')
        )

        path = join(output, f'{timestr}-TransE.pickle')
        checkpoint = torch.load(path)
        entity_path = join(output, 'entity_embedding.npy')
        relation_path = join(output, 'relation_embedding.npy')
        ent_emb = checkpoint['entity_embeddings.weight'].cpu().data.numpy()
        np.save(entity_path, ent_emb)
        rel_emb = checkpoint['relation_embeddings.weight'].cpu().data.numpy()
        np.save(relation_path, rel_emb)

        metrics = compute_metric_results(
            kg_embedding_model=model,
            # mapped_train_triples=train,
            mapped_test_triples=test,
            all_entities=all_entities,
            filter_neg_triples=args.filter_neg_triples,
            device=device
        )

        log_metric("Mean Rank", metrics.mean_rank)
        log_metric("Mean Reciprocal Rank", metrics.mean_reciprocal_rank)

        for k, v in metrics.hits_at_k.items():
            log_metric(f'Hits_at_{k}', v)


if __name__ == "__main__":
    start = time.time()
    main(parse_args())
    end = time.time()
    logging.info('total time: %d', round(end-start))
