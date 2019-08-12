import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class OutputGenerator:

    def __init__(self, output_path):
        self.output_path = output_path
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        matplotlib.use('Agg')

    def create_prediction_tables(self, triples, ranks):
        stop = int(len(ranks) / 2)
        triples_str = [str(tuple(triple)) for triple in triples][:stop]
        prediction_table_head = pd.DataFrame(np.column_stack([triples_str,
                                                              ranks[1::2]]),
                                             columns=['triple',
                                                      'rank of true head'])
        prediction_table_tail = pd.DataFrame(np.column_stack([triples_str,
                                                              ranks[0::2]]),
                                             columns=['triple',
                                                      'rank of true tail'])
        file_name_head = os.path.join(self.output_path,
                                      'prediction_table_head.pkl')
        prediction_table_head.to_pickle(file_name_head)
        file_name_tail = os.path.join(self.output_path,
                                      'prediction_table_tail.pkl')
        prediction_table_tail.to_pickle(file_name_tail)

    def log_top10(self, triples, all_scores):
        for triple, (_, missing_elem, scores) in zip(triples, all_scores):
            if missing_elem == 'head':
                triple_with_gap = tuple('?')+tuple(triple[1:])
            else:
                triple_with_gap = tuple(triple[:-1])+tuple('?')
            print('given: %s' % str(triple_with_gap))

            entity2score = dict()
            for entity_id, score in enumerate(scores):
                entity2score[entity_id] = score
            print('scores for missing element (top 20 entitites):')
            sorted_score = sorted(entity2score, key=entity2score.get,
                                  reverse=True)
            for i in range(10):
                entity = sorted_score[i]
                score = entity2score[entity]
                print('entity-id: %i, score: %f' % (entity, score))

    def compute_metrics(self, ranks):
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

    def find_prediction_examples(self, triples, ranks, good):

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
            if i % 2 == 1:  # head prediciton
                triple2rank_head[str(triples[i])] = ranks[i]
            else:  # tail prediction
                triple2rank_tail[str(triples[i])] = ranks[i]

        if not good:
            # worst head predictions
            worst_predictions_head = get_triples_and_ranks(triple2rank_head,
                                                           not good, 200)
            file_name = os.path.join(self.output_path,
                                     'worst_predictions_head.pkl')
            worst_predictions_head.to_pickle(file_name)

            # worst tail predictions
            worst_predictions_tail = get_triples_and_ranks(triple2rank_tail,
                                                           not good, 200)
            file_name = os.path.join(self.output_path,
                                     'worst_predictions_tail.pkl')
            worst_predictions_tail.to_pickle(file_name)

        else:
            # best head predictions
            best_predictions_head = get_triples_and_ranks(triple2rank_head,
                                                          not good, 200)
            file_name = os.path.join(self.output_path,
                                     'best_predictions_head.pkl')
            best_predictions_head.to_pickle(file_name)

            # best tail predictions
            best_predictions_tail = get_triples_and_ranks(triple2rank_tail,
                                                          not good, 200)
            file_name = os.path.join(self.output_path,
                                     'best_predictions_tail.pkl')
            best_predictions_tail.to_pickle(file_name)

    def compare_head_tail_predictions(self, ranks):
        metrics = ['MR', 'MRR', 'HITS@1', 'HITS@3', 'HITS@10']
        heads_and_tails = [round(self.compute_metrics(ranks)[metric], 3)
                           for metric in metrics]
        only_tails = [round(self.compute_metrics(ranks[0::2])[metric], 3)
                      for metric in metrics]
        only_heads = [round(self.compute_metrics(ranks[1::2])[metric], 3)
                      for metric in metrics]

        compare_head_tail_df = pd.DataFrame(np.column_stack([metrics,
                                                             heads_and_tails,
                                                             only_heads,
                                                             only_tails]),
                                            columns=['metric',
                                                     'heads and tails',
                                                     'only heads',
                                                     'only tails'])

        file_name = os.path.join(self.output_path, 'compare_head_tail.pkl')
        compare_head_tail_df.to_pickle(file_name)

    def create_average_position_table(self, avg_position_subjects,
                                      avg_position_objects):
        entity_ids = []
        avg_positions = []
        for entity_id in sorted(avg_position_subjects,
                                key=avg_position_subjects.get):
            entity_ids.append(entity_id)
            avg_positions.append(round(avg_position_subjects[entity_id], 3))

        avg_pos_table_head = pd.DataFrame(np.column_stack([entity_ids,
                                                           avg_positions]),
                                          columns=['entity_id',
                                                   'average position'])
        file_name_head = os.path.join(self.output_path,
                                      'avg_position_table_head.pkl')
        avg_pos_table_head['entity_id'] = avg_pos_table_head['entity_id'].astype(int)  # noqa
        avg_pos_table_head.to_pickle(file_name_head)

        entity_ids = []
        avg_positions = []
        for entity_id in sorted(avg_position_objects,
                                key=avg_position_objects.get):
            entity_ids.append(entity_id)
            avg_positions.append(round(avg_position_objects[entity_id], 3))

        avg_pos_table_tail = pd.DataFrame(np.column_stack([entity_ids,
                                                           avg_positions]),
                                          columns=['entity_id',
                                                   'average position'])
        file_name_tail = os.path.join(self.output_path,
                                      'avg_position_table_tail.pkl')
        avg_pos_table_tail['entity_id'] = avg_pos_table_tail['entity_id'].astype(int)  # noqa
        avg_pos_table_tail.to_pickle(file_name_tail)

    def create_highly_ranked_count_tables(self, count_subjects, count_objects):
        entity_ids = []
        counts = []
        for entity_id in sorted(count_subjects, key=count_subjects.get,
                                reverse=True):
            entity_ids.append(entity_id)
            counts.append(count_subjects[entity_id])

        highly_ranked_table_head = pd.DataFrame(np.column_stack([entity_ids,
                                                                 counts]),
                                                columns=['entity_id', 'count'])
        file_name_head = os.path.join(self.output_path,
                                      'highly_ranked_table_head.pkl')
        highly_ranked_table_head['entity_id'] = highly_ranked_table_head['entity_id'].astype(int)  # noqa
        highly_ranked_table_head['count'] = highly_ranked_table_head['count'].astype(int)  # noqa
        highly_ranked_table_head.to_pickle(file_name_head)

        entity_ids = []
        counts = []
        for entity_id in sorted(count_objects, key=count_objects.get,
                                reverse=True):
            entity_ids.append(entity_id)
            counts.append(count_objects[entity_id])

        highly_ranked_table_tail = pd.DataFrame(np.column_stack([entity_ids,
                                                                 counts]),
                                                columns=['entity_id', 'count'])
        file_name_tail = os.path.join(self.output_path,
                                      'highly_ranked_table_tail.pkl')
        highly_ranked_table_tail['entity_id'] = highly_ranked_table_tail['entity_id'].astype(int)  # noqa
        highly_ranked_table_tail['count'] = highly_ranked_table_tail['count'].astype(int)  # noqa
        highly_ranked_table_tail.to_pickle(file_name_tail)

    def create_top_ranked_count_tables(self, count_subjects, count_objects):
        entity_ids = []
        counts = []
        for entity_id in sorted(count_subjects, key=count_subjects.get,
                                reverse=True):
            entity_ids.append(entity_id)
            counts.append(count_subjects[entity_id])

        top_ranked_table_head = pd.DataFrame(np.column_stack([entity_ids,
                                                              counts]),
                                             columns=['entity_id', 'count'])
        file_name_head = os.path.join(self.output_path,
                                      'top_ranked_table_head.pkl')
        top_ranked_table_head['entity_id'] = top_ranked_table_head['entity_id'].astype(int)  # noqa
        top_ranked_table_head['count'] = top_ranked_table_head['count'].astype(int)  # noqa
        top_ranked_table_head.to_pickle(file_name_head)

        entity_ids = []
        counts = []
        for entity_id in sorted(count_objects, key=count_objects.get,
                                reverse=True):
            entity_ids.append(entity_id)
            counts.append(count_objects[entity_id])

        top_ranked_table_tail = pd.DataFrame(np.column_stack([entity_ids,
                                                              counts]),
                                             columns=['entity_id', 'count'])
        file_name_tail = os.path.join(self.output_path,
                                      'top_ranked_table_tail.pkl')
        top_ranked_table_tail['entity_id'] = top_ranked_table_tail['entity_id'].astype(int)  # noqa
        top_ranked_table_tail['count'] = top_ranked_table_tail['count'].astype(int)  # noqa
        top_ranked_table_tail.to_pickle(file_name_tail)

    def plot_scores_of_one_prediction(self, scores):
        x = np.arange(len(scores))
        plt.figure()
        plt.bar(x, sorted(scores.cpu(), reverse=True), align='center')
        plt.xlabel('Entity')
        plt.ylabel('Score')
        plt.title('scores of one prediction')
        plt.savefig(os.path.join(self.output_path, 'scores.png'))

        # additionally create CDF plot
        plt.figure()
        plt.xlabel('Entity')
        plt.ylabel('Score')
        plt.title('scores of one prediction')
        plt.plot(np.linspace(0, 1, num=len(scores)),
                 sorted(scores.cpu(), reverse=True))
        plt.xticks([])
        plt.savefig(os.path.join(self.output_path, 'scores_cdf.png'))
