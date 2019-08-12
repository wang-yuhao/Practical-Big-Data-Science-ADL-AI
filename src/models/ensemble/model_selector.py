"""Model selector class and related utilites"""
import torch
import numpy as np
import tqdm
from src.models.api import EvaluationModel

def crop_per_relation(source: np.array, limit=100):
    """
    Returns a dict contains np.array of triples per relation
    """
    np.random.shuffle(source)
    triples = dict()
    counter = dict()
    for src in source:
        curr_rel = src[1]
        if curr_rel not in counter:
            counter[curr_rel] = 0
        if counter[curr_rel] == limit:
            continue

        if curr_rel not in triples:
            triples[curr_rel] = []
        triples[curr_rel].append(src)
        counter[curr_rel] += 1

    # convert lists to np.arrays
    triples = {k: np.asarray(v) for k, v in triples.items()}
    return triples

def evaluation_per_relation(triples: dict, model: EvaluationModel, batch_size: int = 4):
    """
    :param triples: It should be a  dict in form (Relation id):[(s_1,p_1,o_1)...(s_n,p_n,o_n)]
    """
    # Evaluate per relation and store scores/evaluation measures
    score_per_rel = dict()

    for k in tqdm.tqdm(triples.keys()):
        # use API to evaluate model and generate model output for error analysis
        sub = torch.tensor(triples[k][:, 0]).cuda()
        pra = torch.tensor(triples[k][:, 1]).cuda()
        obj = torch.tensor(triples[k][:, 2]).cuda()
        score_per_rel[k] = model.evaluate_only_metrics(sub, pra, obj, batch_size=batch_size)

    return score_per_rel


def generate_lookup(models: list, triples: np.array,
                    num_samples: int = 100, batch_size=4):
    """
    Generate lookup by evaluating models per relation

    It will be performed on TRAIN set for fariness(and
    not all relations are present in test set. 12 of them are missing)
    """
    lookup = dict()
    np.random.seed(0)
    triples = crop_per_relation(triples, num_samples)

    # Get scores for all models
    model_scores = dict()
    model_names = []
    for (model_name, model) in list(models.items()):
        print(model_name)
        model_names.append(model_name)
        model_scores[model_name] = evaluation_per_relation(triples, model, batch_size)

    # Aggregate scores into lookup table
    # Select the model with best MRR
    for i in range(len(triples.keys())):
        lookup[i] = max(model_scores, key=lambda x, curr_rel=i: model_scores[x][curr_rel]['MRR'])

    return lookup


class ModelSelector(EvaluationModel):
    """Emsemble model with choose model by given relation with performance of model"""
    def __init__(self, models, neg_sample_generator, lookup):
        """
        init model selector with list of models and lookup
        :param models: list of models
        :param lookup: A dict with (relation id, best_performed_model)
        """
        super(ModelSelector, self).__init__(None, neg_sample_generator)
        self.models = models
        self.lookup = lookup


    def split_batch_with_lookup(self, sub: torch.tensor, pra: torch.tensor,
                                obj: torch.tensor) -> dict():
        """Splits batchs per relation with same lookup table value"""
        model_batches = dict()

        for (model_name, _) in list(self.models.items()):
            model_batches[model_name] = dict()
            mask = [self.lookup[elem] == model_name for elem in pra.tolist()]
            model_batches[model_name]['s'] = sub[mask]
            model_batches[model_name]['p'] = pra[mask]
            model_batches[model_name]['o'] = obj[mask]

        return model_batches


    def predict_object_scores(self, s: torch.tensor, p: torch.tensor,
                              o: torch.tensor) -> torch.tensor:
        """
        Link Prediction (right-sided)

        :param s: torch.tensor, dtype: int, shape: (batch_size,)
            The subjects' IDs.
        :param p: torch.tensor, dtype: int, shape: (batch_size,)
            The predicates' IDs
        :param o: torch.tensor, dtype: int, shape: (batch_size,)
            The objects' IDs.

        :return: torch.tensor, dtype: float, shape: (batch_size, num_entities)
        """
        model_batches = self.split_batch_with_lookup(s, p, o)
        outputs = []
        #Call predict funcion of model
        for (model_name, model) in list(self.models.items()):
            model.batch_size = list(model_batches[model_name]['s'].size())[0]
            if model.batch_size == 0:
                continue
            model_output = model.predict_object_scores(model_batches[model_name]['s'],
                                                       model_batches[model_name]['p'],
                                                       model_batches[model_name]['o'])
            outputs.append(model_output)

        return torch.cat(outputs)


    def predict_subject_scores(self, s: torch.tensor, p: torch.tensor,
                               o: torch.tensor) -> torch.tensor:
        """
        Link Prediction (left-sided)

        :param s: torch.tensor, dtype: int, shape: (batch_size,)
            The subjects' IDs.
        :param p: torch.tensor, dtype: int, shape: (batch_size,)
            The predicates' IDs
        :param o: torch.tensor, dtype: int, shape: (batch_size,)
            The objects' IDs.

        :return: torch.tensor, dtype: float, shape: (batch_size, num_entities)
        """
        model_batches = self.split_batch_with_lookup(s, p, o)
        outputs = []
        for (model_name, model) in list(self.models.items()):
            model.batch_size = list(model_batches[model_name]['s'].size())[0]
            if model.batch_size == 0:
                continue
            model_output = model.predict_subject_scores(model_batches[model_name]['s'],
                                                        model_batches[model_name]['p'],
                                                        model_batches[model_name]['o'])
            outputs.append(model_output)

        return torch.cat(outputs)
