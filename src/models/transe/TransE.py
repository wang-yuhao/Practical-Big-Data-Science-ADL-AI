"""
Design TransE model and validate function

Interface: 'TransE', 'compute_metric_results'
"""
import logging
import timeit
from dataclasses import dataclass
from typing import Mapping, Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.autograd
from torch import nn
import tqdm
from tqdm import trange
import random
from tqdm import tqdm


import torch.optim as optim

EMOJI = '⚽️'
DEFAULT_HITS_AT_K = [1, 3, 5, 10]

__all__ = ['TransE', 'compute_metric_results']

LOG = logging.getLogger(__name__)

class TransE(nn.Module):
    """
    The class of TransE model, definied by paper [Translating Embeddings for Modeling
    Multi-relational Data](Bordes et al. 2013)

    Inherit nn.Module class
    """
    def __init__(self,
	         margin_loss: float,
             embedding_dim: int,
             scoring_function: Optional[int] = 1,
             normalization_of_entities: Optional[int] = 2,
             random_seed: Optional[int] = None,
             preferred_device: str = 'cpu',
             ) -> None:
        super().__init__()

        # Device selection
        self._get_device(preferred_device)

        self.random_seed = random_seed

        # Random seeds have to set before the embeddings are initialized
        if self.random_seed is not None:
            np.random.seed(seed=self.random_seed)
            torch.manual_seed(seed=self.random_seed)
            random.seed(self.random_seed)

        # Loss
        self.margin_loss = margin_loss
        self.criterion = nn.MarginRankingLoss(
            margin=self.margin_loss,
            # size_average=self.margin_ranking_loss_size_average,
        )

        # Entity dimension
        # The number of entities in the knowledge graph
        self.num_entities = None
        # The number of unique relation types in the knowledge graph
        self.num_relations = None
        # The dimension of the embeddings to generate
        self.embedding_dim = embedding_dim

        # Default optimizer for all classes
        self.optimizer = None
        self.default_optimizer = optim.SGD

        # Instance attributes that are defined when calling other functions
        # Calling data load function

        # Calling fit function
        self.entity_embeddings = None
        self.relation_embeddings = None

        self.learning_rate = None
        self.num_epochs = None
        self.batch_size = None

        # The type of normalization for entities
        self.l_p_norm_entities = normalization_of_entities
        self.scoring_fct_norm = scoring_function


    def _initialize(self):

        embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

        norms = torch.norm(self.relation_embeddings.weight, p=2, dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

    def __init_subclass__(cls):  # noqa: D105
        if not getattr(cls, 'model_name', None):
            raise TypeError('missing model_name class attribute')

    def _get_entity_embeddings(self, entities):
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)

    def _get_device(self, device: str = 'cpu') -> None:
        """Get the Torch device to use."""
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
                print('No cuda devices were available. The model runs on CPU')
        else:
            self.device = torch.device('cpu')

    def _to_cpu(self):
        """Transfer the entire model to CPU"""
        self._get_device('cpu')
        self.to(self.device)
        torch.cuda.empty_cache()

    def _to_gpu(self):
        """Transfer the entire model to GPU"""
        self._get_device('gpu')
        self.to(self.device)
        torch.cuda.empty_cache()

    def _compute_loss(self, positive_scores: torch.Tensor, negative_scores:
		      torch.Tensor)-> torch.Tensor:
        tensor = torch.FloatTensor([-1])
        tensor = tensor.expand(positive_scores.shape[0]).to(self.device)
        loss = self.criterion(positive_scores, negative_scores, tensor)
        return loss

    def _init_embeddings(self):
        """initialize embedding"""
        self.entity_embeddings = nn.Embedding(
            self.num_entities,
            self.embedding_dim,
            # max_norm=self.entity_embedding_max_norm,
            norm_type=self.l_p_norm_entities,
            )
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self._initialize()

    def fit(self,
            pos_triples: np.ndarray,
            learning_rate: float,
            num_epochs: int,
            batch_size: int,
            optimizer: Optional[torch.optim.Optimizer] = None,
            weight_decay: Optional[float] = 0,
            tqdm_kwargs: Optional[Mapping[str, Any]] = None,
            ) -> List[float]:
        """fit the embedding"""
        LOG.info('initialize embedding')
        self._init_embeddings()

        self.to(self.device)

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        if optimizer is None:
            optimizer = self.default_optimizer
        self.optimizer = optimizer(self.parameters(),
		                   lr=self.learning_rate,
		                   weight_decay=weight_decay)

        LOG.info('****Run Model On %d****', str(self.device).upper())

        loss_per_epoch = []
        num_pos_triples = pos_triples.shape[0]
        all_entities = np.arange(self.num_entities)

        epoch_loss = None

        _tqdm_kwargs = dict(desc='Training epoch')
        if tqdm_kwargs:
            _tqdm_kwargs.update(tqdm_kwargs)

        indices = np.arange(num_pos_triples)
        num_positives = self.batch_size
        trange_bar = trange(self.num_epochs, **_tqdm_kwargs)
        for _ in trange_bar:
            np.random.shuffle(indices)
            pos_triples = pos_triples[indices]
            pos_batches = _split_list_in_batches(input_list=pos_triples,
			                         batch_size=num_positives)

            current_epoch_loss = 0.

            for _, pos_batch in enumerate(pos_batches):

                current_batch_size = len(pos_batch)

                batch_subjs, batch_relations, batch_objs = slice_triples(pos_batch)

                num_subj_corrupt = len(pos_batch) // 2
                num_obj_corrupt = len(pos_batch) - num_subj_corrupt
                pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=self.device)

                corrupted_subj_indices = np.random.choice(np.arange(0, self.num_entities),
                                size=num_subj_corrupt)
                corrupted_subjects = np.reshape(all_entities[corrupted_subj_indices], 
				                newshape=(-1, 1))
                subject_based_corrupted_triples = np.concatenate([
corrupted_subjects, batch_relations[:num_subj_corrupt],batch_objs[:num_subj_corrupt]], axis=1)
                corrupted_obj_indices = np.random.choice(np.arange(0, self.num_entities),
				                size=num_obj_corrupt)
                corrupted_objects = np.reshape(all_entities[corrupted_obj_indices], 
				                newshape=(-1, 1))
                object_based_corrupted_triples = np.concatenate([
					            batch_subjs[num_subj_corrupt:],batch_relations[num_subj_corrupt:],
								corrupted_objects], axis=1)

                neg_batch = np.concatenate([
					subject_based_corrupted_triples, 
					object_based_corrupted_triples], axis=0)
                neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=self.device)

                self.optimizer.zero_grad()

                loss = self(pos_batch, neg_batch)
                current_epoch_loss += (loss.item() * current_batch_size)

                loss.backward()
                self.optimizer.step()

            # track epoch loss
            previous_loss = epoch_loss
            epoch_loss = current_epoch_loss / (len(pos_triples) *\
			                            self.entity_embeddings.num_embeddings)
            loss_per_epoch.append(epoch_loss)
            trange_bar.set_postfix(loss=epoch_loss, previous_loss=previous_loss)

        LOG.info("Training took {str(round(stop_training - start_training))} seconds \n")

        return loss_per_epoch

    def predict(self, triples):
        """predict the score of triples"""
        # Check if the model has been fitted yet.
        if self.entity_embeddings is None:
            print('The model has not been fitted yet. Predictions are based on\
			        randomly initialized embeddings.')
            self._init_embeddings()
        
        triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    # pylint: disable=arguments-differ
    def forward(self, batch_positives, batch_negatives):
        """override forward"""
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)
        loss = self._compute_loss(
            positive_scores=positive_scores,
            negative_scores=negative_scores)
        return loss


    def _score_triples(self, triples):
        """prepare the triples to compute their score"""
        head_embeddings, relation_embeddings, tail_embeddings =\
            self._get_triple_embeddings(triples)
        scores = self._compute_scores(head_embeddings, relation_embeddings, tail_embeddings)
        return scores

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings):
        """Compute the scores based on the head, relation, and tail embeddings.
        :param head_embeddings: embeddings of head entities of dimension batchsize x embedding_dim
        :param relation_embeddings: emebddings of relation embeddings of dimension
		    batchsize x embedding_dim
        :param tail_embeddings: embeddings of tail entities of dimension batchsize x embedding_dim
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        # Add the vector element wise
        sum_res = head_embeddings + relation_embeddings - tail_embeddings
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        return distances

    def _get_triple_embeddings(self, triples):
        """convert the triples to embeddings"""
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_relation_embeddings(relations),
            self._get_entity_embeddings(tails),
            )

    def _get_relation_embeddings(self, relations):
        """convert the relations to embeddings"""
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)

def slice_triples(triples: np.ndarray):
    """Get the heads, relations, and tails from a matrix of triples."""
    head = triples[:, 0:1]
    relation = triples[:, 1:2]
    tail = triples[:, 2:3]
    return head, relation, tail

def _split_list_in_batches(input_list: np.ndarray, batch_size: int):
    """split list in batches"""
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

# metrics computation

def _hash_triples(triples: Iterable[Hashable]) -> int:
    """Hash a list of triples."""
    return hash(tuple(triples))


def update_hits_at_k(
        hits_at_k_values: Dict[int, List[float]],
        rank_of_positive_subject_based: int,
        rank_of_positive_object_based: int
) -> None:
    """Update the Hits@K dictionary for two values."""
    for k, values in hits_at_k_values.items():
        if rank_of_positive_subject_based < k:
            values.append(1.0)
        else:
            values.append(0.0)

        if rank_of_positive_object_based < k:
            values.append(1.0)
        else:
            values.append(0.0)


def _create_corrupted_triples(triple, all_entities):
    """create corrupted triples with all entities"""
    candidate_entities_subject_based = all_entities[all_entities != triple[0:1]].reshape((-1, 1))
    candidate_entities_object_based = all_entities[all_entities != triple[2:3]].reshape((-1, 1))

    # Extract current test tuple: Either (subject,predicate) or (predicate,object)
    tuple_subject_based = np.reshape(a=triple[1:3], newshape=(1, 2))
    tuple_object_based = np.reshape(a=triple[0:2], newshape=(1, 2))

    # Copy current test tuple
    tuples_subject_based = np.repeat(
        a=tuple_subject_based,
        repeats=candidate_entities_subject_based.shape[0],
        axis=0)
    tuples_object_based = np.repeat(
        a=tuple_object_based,
        repeats=candidate_entities_object_based.shape[0],
        axis=0)

    corrupted_subject_based = np.concatenate(
        [candidate_entities_subject_based, tuples_subject_based], axis=1)

    corrupted_object_based = np.concatenate(
        [tuples_object_based, candidate_entities_object_based], axis=1)

    return corrupted_subject_based, corrupted_object_based


def _filter_corrupted_triples(
        corrupted_subject_based,
        corrupted_object_based,
        all_pos_triples_hashed,
):
    """filter the positive triples from corrupted triples"""
    corrupted_subject_based_hashed = np.apply_along_axis(
        _hash_triples, 1, corrupted_subject_based)
    mask = np.in1d(corrupted_subject_based_hashed, all_pos_triples_hashed, invert=True)
    mask = np.where(mask)[0]
    corrupted_subject_based = corrupted_subject_based[mask]

    corrupted_object_based_hashed = np.apply_along_axis(_hash_triples, 1, corrupted_object_based)
    mask = np.in1d(corrupted_object_based_hashed, all_pos_triples_hashed, invert=True)
    mask = np.where(mask)[0]

    if mask.size == 0:
        raise Exception("User selected filtered metric computation, but all corrupted triples \
                        exists,also a positive triples.")
    corrupted_object_based = corrupted_object_based[mask]

    return corrupted_subject_based, corrupted_object_based


def _compute_filtered_rank(
        kg_embedding_model,
        pos_triple,
        corrupted_subject_based,
        corrupted_object_based,
        device,
        all_pos_triples_hashed,
) -> Tuple[int, int]:
    """
    :param kg_embedding_model:
    :param pos_triple:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples_hashed:
    """
    corrupted_subject_based, corrupted_object_based = _filter_corrupted_triples(
        corrupted_subject_based=corrupted_subject_based,
        corrupted_object_based=corrupted_object_based,
        all_pos_triples_hashed=all_pos_triples_hashed)

    return _compute_rank(
        kg_embedding_model=kg_embedding_model,
        pos_triple=pos_triple,
        corrupted_subject_based=corrupted_subject_based,
        corrupted_object_based=corrupted_object_based,
        device=device,
    )


def _compute_rank(
        kg_embedding_model,
        pos_triple,
        corrupted_subject_based,
        corrupted_object_based,
        device,
) -> Tuple[int, int]:
    """
    :param kg_embedding_model:
    :param pos_triple:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples_hashed: This parameter isn't used but is necessary for compatability
    """

    corrupted_subject_based = torch.tensor(
        corrupted_subject_based,
        dtype=torch.long,
        device=device
        )
    corrupted_object_based = torch.tensor(corrupted_object_based, dtype=torch.long, device=device)

    scores_of_corrupted_subjects = kg_embedding_model.predict(corrupted_subject_based)
    scores_of_corrupted_objects = kg_embedding_model.predict(corrupted_object_based)

    pos_triple = np.array(pos_triple)
    pos_triple = np.expand_dims(a=pos_triple, axis=0)
    pos_triple = torch.tensor(pos_triple, dtype=torch.long, device=device)

    score_of_positive = kg_embedding_model.predict(pos_triple)

    scores_subject_based = np.append(arr=scores_of_corrupted_subjects, values=score_of_positive)
    indice_of_pos_subject_based = scores_subject_based.size - 1

    scores_object_based = np.append(arr=scores_of_corrupted_objects, values=score_of_positive)
    indice_of_pos_object_based = scores_object_based.size - 1

    _, sorted_score_indices_subject_based = torch.sort(
torch.tensor(scores_subject_based, dtype=torch.float),
        descending=False)
    sorted_score_indices_subject_based = sorted_score_indices_subject_based.cpu().numpy()

    _, sorted_score_indices_object_based = torch.sort(
torch.tensor(scores_object_based, dtype=torch.float),
        descending=False)
    sorted_score_indices_object_based = sorted_score_indices_object_based.cpu().numpy()

    # Get index of first occurrence that fulfills the condition
    rank_of_positive_subject_based = np.where(sorted_score_indices_subject_based == \
	                                    indice_of_pos_subject_based)[0][0]
    rank_of_positive_object_based = np.where(sorted_score_indices_object_based == \
	                                    indice_of_pos_object_based)[0][0]

    return (
        rank_of_positive_subject_based,
        rank_of_positive_object_based,
    )


@dataclass
class MetricResults:
    """Results from computing metrics."""

    mean_rank: float
    mean_reciprocal_rank: float
    hits_at_k: Dict[int, float]


def compute_metric_results(
        all_entities,
        kg_embedding_model,
        # mapped_train_triples,
        mapped_test_triples,
        device,
        filter_neg_triples=True,
        ks: Optional[List[int]] = None,
        *,
        use_tqdm: bool = True,
) -> MetricResults:
    """Compute the metric results.
    :param all_entities:
    :param kg_embedding_model:
    :param mapped_train_triples:
    :param mapped_test_triples:
    :param device:
    :param filter_neg_triples:
    :param ks:
    :param use_tqdm: Should a progress bar be shown?
    :return:
    """
    start = timeit.default_timer()

    ranks: List[int] = []
    hits_at_k_values = {
        k: []
        for k in (ks or DEFAULT_HITS_AT_K)
    }
    kg_embedding_model = kg_embedding_model.eval()
    kg_embedding_model = kg_embedding_model.to(device)

    # all_pos_triples = np.concatenate([mapped_train_triples, mapped_test_triples], axis=0)
    # all_pos_triples_hashed = np.apply_along_axis(_hash_triples, 1, all_pos_triples)

    compute_rank_fct: Callable[..., Tuple[int, int]] = (
        _compute_filtered_rank
        if filter_neg_triples else
        _compute_rank
    )

    if use_tqdm:
        mapped_test_triples = tqdm(mapped_test_triples, desc=f'{EMOJI} corrupting triples')
    for pos_triple in mapped_test_triples:
        corrupted_subject_based, corrupted_object_based = _create_corrupted_triples(
            triple=pos_triple,
            all_entities=all_entities,
        )

        rank_of_positive_subject_based, rank_of_positive_object_based = compute_rank_fct(
            kg_embedding_model=kg_embedding_model,
            pos_triple=pos_triple,
            corrupted_subject_based=corrupted_subject_based,
            corrupted_object_based=corrupted_object_based,
            device=device,
            # all_pos_triples_hashed=all_pos_triples_hashed,
        )

        ranks.append(rank_of_positive_subject_based)
        ranks.append(rank_of_positive_object_based)

        # Compute hits@k for k in {1,3,5,10}
        update_hits_at_k(
            hits_at_k_values,
            rank_of_positive_subject_based=rank_of_positive_subject_based,
            rank_of_positive_object_based=rank_of_positive_object_based,
        )

    mean_rank = float(np.mean(ranks))
    while 0 in ranks:
        ranks.remove(0)
        ranks.append(10000000000)

    mean_reciprocal_rank = float((1/np.vstack(ranks)).mean())
    hits_at_k: Dict[int, float] = {
        k: np.mean(values)
        for k, values in hits_at_k_values.items()
    }

    stop = timeit.default_timer()
    LOG.info("Evaluation took %.2fs seconds", stop - start)

    return MetricResults(
        mean_rank=mean_rank,
        mean_reciprocal_rank=mean_reciprocal_rank,
        hits_at_k=hits_at_k,
    )
