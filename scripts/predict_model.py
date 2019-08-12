import argparse
import logging
import random
import pickle
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models.model import KGEModel


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Load trained model and use it for predictions',
        usage='predict_model.py [<args>]'
    )
    parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('-d', '--data', type=str, default=None)
    parser.add_argument('-r', '--raw', type=bool, default=True)
    parser.add_argument('-dim', '--hidden_dim', default=1000, type=int)

    return parser.parse_args(args)


def load_data(data_path):
    path_entity2id = os.path.join(data_path, 'entity_to_id.pickle')
    with open(path_entity2id, "rb") as handle:
        entity2id = pickle.load(handle)

    path_relation2id = os.path.join(data_path, 'relation_to_id.pickle')
    with open(path_relation2id, "rb") as handle:
        relation2id = pickle.load(handle)

    path_test = os.path.join(data_path, 'test.pickle')
    with open(path_test, 'rb') as handle:
        test_triples = pickle.load(handle)

    return entity2id, relation2id, test_triples


def log_top10(triples, metrics, all_scores):
    for metric in metrics:
        logging.info('%s: %f', metric, metrics[metric])

    for triple, (_, missing_elem, scores) in zip(triples, all_scores):
        if missing_elem == 'head':
            triple_with_gap = tuple('?')+tuple(triple[1:])
        else:
            triple_with_gap = tuple(triple[:-1])+tuple('?')
        logging.info('given: %s', str(triple_with_gap))

        entity2score = dict()
        for entity_id, score in enumerate(scores):
            entity2score[entity_id] = score
        logging.info('scores for missing element (top 20 entitites):')
        sorted_score = sorted(entity2score, key=entity2score.get, reverse=True)
        for i in range(20):
            entity = sorted_score[i]
            score = entity2score[entity]
            logging.info('entity-id: %i, score: %f', entity, score)
        print()


def create_prediction_tables(triples, ranks, save_path):
    middle = int(len(ranks)/2)
    assert middle == len(triples)
    triples_str = [str(tuple(triple)) for triple in triples]
    prediction_table_head = pd.DataFrame(np.column_stack([triples_str,
                                                          ranks[:middle]]),
                                         columns=['triple',
                                                  'rank of true head'])
    prediction_table_tail = pd.DataFrame(np.column_stack([triples_str,
                                                          ranks[middle:]]),
                                         columns=['triple',
                                                  'rank of true tail'])
    file_name_head = os.path.join(save_path, 'prediction_table_head.pkl')
    prediction_table_head.to_pickle(file_name_head)
    file_name_tail = os.path.join(save_path, 'prediction_table_tail.pkl')
    prediction_table_tail.to_pickle(file_name_tail)


def compute_metrics(ranks):
    logs = []
    for rank in ranks:
        logs.append({
            'MRR': 1.0/rank,
            'MR': float(rank),
            'HITS@1': 1.0 if rank <= 1 else 0.0,
            'HITS@3': 1.0 if rank <= 3 else 0.0,
            'HITS@10': 1.0 if rank <= 10 else 0.0,
        })
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    return metrics


def plot_scores_of_one_prediction(scores, save_path):
    y_pos = np.arange(len(scores))
    plt.figure()
    plt.bar(y_pos, sorted(scores.cpu(), reverse=True), align='center')
    plt.xlabel('Entity')
    plt.ylabel('Score')
    plt.title('scores of one prediction')
    plt.savefig(os.path.join(save_path, 'scores.png'))

    # additionally create CDF plot
    plt.figure()
    plt.xlabel('Entity')
    plt.ylabel('Score')
    plt.title('scores of one prediction')
    plt.plot(np.linspace(0, 1, num=len(scores)),
             sorted(scores.cpu(), reverse=True))
    plt.xticks([])
    plt.savefig(os.path.join(save_path, 'scores_cdf.png'))


def find_prediction_examples(triples, ranks, good, save_path):

    def get_triples_and_ranks(triple2rank, reverse, num):
        sorted_triples = sorted(triple2rank, key=triple2rank.get,
                                reverse=reverse)[:num]
        ranks = [triple2rank[triple] for triple in sorted_triples]
        return pd.DataFrame(np.column_stack([sorted_triples, ranks]),
                            columns=['triple', 'rank'])

    triples = np.append(triples, triples, axis=0)  # head / tail
    triple2rank_head = {}
    triple2rank_tail = {}

    for i in range(len(ranks)):
        if i < len(ranks)/2:  # head prediciton
            triple2rank_head[str(triples[i])] = ranks[i]
        else:  # tail prediction
            triple2rank_tail[str(triples[i])] = ranks[i]

    if not good:
        # worst head predictions
        worst_predictions_head = get_triples_and_ranks(triple2rank_head,
                                                       not good, 10)
        file_name = os.path.join(save_path, 'worst_predictions_head.pkl')
        worst_predictions_head.to_pickle(file_name)

        # worst tail predictions
        worst_predictions_tail = get_triples_and_ranks(triple2rank_tail,
                                                       not good, 10)
        file_name = os.path.join(save_path, 'worst_predictions_tail.pkl')
        worst_predictions_tail.to_pickle(file_name)

    else:
        # best head predictions
        best_predictions_head = get_triples_and_ranks(triple2rank_head,
                                                      not good, 10)
        file_name = os.path.join(save_path, 'best_predictions_head.pkl')
        best_predictions_head.to_pickle(file_name)

        # best tail predictions
        best_predictions_tail = get_triples_and_ranks(triple2rank_tail,
                                                      not good, 10)
        file_name = os.path.join(save_path, 'best_predictions_tail.pkl')
        best_predictions_tail.to_pickle(file_name)


def compare_head_tail_predictions(ranks, save_path):
    metrics = ['MR', 'MRR', 'HITS@1', 'HITS@3', 'HITS@10']
    heads_and_tails = [round(compute_metrics(ranks)[metric], 3)
                       for metric in metrics]
    only_heads = [round(compute_metrics(ranks[:int(len(ranks)/2)])[metric], 3)
                  for metric in metrics]
    only_tails = [round(compute_metrics(ranks[int(len(ranks)/2):])[metric], 3)
                  for metric in metrics]

    compare_head_tail_df = pd.DataFrame(np.column_stack([metrics,
                                                         heads_and_tails,
                                                         only_heads,
                                                         only_tails]),
                                        columns=['metric', 'heads and tails',
                                                 'only heads', 'only tails'])

    file_name = os.path.join(save_path, 'compare_head_tail.pkl')
    compare_head_tail_df.to_pickle(file_name)


def set_logger(args):
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.model, 'predict.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def main(args):
    """
    Load trained model and use it for predictions.
    """

    if args.model is None or args.data is None:
        raise ValueError('You have to specify model and data input paths.')

    set_logger(args)

    model_path = os.path.join(args.model, 'checkpoint')
    input_path = args.data

    loaded_data = load_data(input_path)
    entity2id = loaded_data[0]
    relation2id = loaded_data[1]
    test_triples = loaded_data[2]

    nentity = len(entity2id)
    nrelation = len(relation2id)

    kge_model = KGEModel(  # default values used
        model_name='RotatE',
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=12.0,
        double_entity_embedding=True,
        # double_relation_embedding=args.double_relation_embedding
    )

    checkpoint = torch.load(model_path)
    kge_model.load_state_dict(checkpoint['model_state_dict'])

    # path = os.path.join(args.model, 'entity_embedding.npy')
    # new_entity_embedding = Parameter(torch.from_numpy(np.load(path)))
    # kge_model.change_entity_embedding(new_entity_embedding)

    # path = os.path.join(args.model, 'relation_embedding.npy')
    # new_relation_embedding = Parameter(torch.from_numpy(np.load(path)))
    # kge_model.change_relation_embedding(new_relation_embedding)

    kge_model.cuda()
    kge_model.eval()

    all_true_triples = []
    args_kge = Namespace(nentity=nentity, nrelation=nrelation,
                         test_batch_size=4, cpu_num=10, cuda=True,
                         test_log_steps=1000)  # default values

    _, _, scores, ranks = kge_model.test_step(kge_model, test_triples,
                                                    all_true_triples, args_kge,
                                                    save_scores=True)

    # generate output for error analysis

    # create tables with rank per triple
    create_prediction_tables(test_triples, ranks, args.model)

    # create bar chart and CDF plot of scores (for one random prediction)
    random_index = random.randint(0, len(scores)-1)
    plot_scores_of_one_prediction(scores[random_index][2],
                                  args.model)

    # find examples of good and bad predictions
    find_prediction_examples(test_triples, ranks, good=True,
                             save_path=args.model)
    find_prediction_examples(test_triples, ranks, good=False,
                             save_path=args.model)

    # compute metrics (MR, MRR and H@N) for head / tail separately
    compare_head_tail_predictions(ranks, save_path=args.model)

    # log_top10(test_triples, metrics, scores)


if __name__ == '__main__':
    main(parse_args())
