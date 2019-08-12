import logging
import torch
import tqdm
import timeit

import warnings

import numpy as np
import torch.nn.functional as F

from typing import Dict, Any, List, Optional, Union, Tuple
from numpy import random

from src.models.api import AbstractModel, EvaluationModel
from src.metrics import Metric
from src.callbacks import Callback
from src.data.utils import split_list_in_batches

logger = logging.getLogger(__name__)


class ConvE(AbstractModel):
    """An implementation of ConvE [dettmers2017]_.
    .. [dettmers2017] Dettmers, T., *et al.* (2017)
    `Convolutional 2d knowledge graph embeddings
    <https://arxiv.org/pdf/1707.01476.pdf>`_. arXiv preprint arXiv:1707.01476.
    .. seealso:: https://github.com/TimDettmers/ConvE/blob/master/model.py
    """

    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 ConvE_input_channels,
                 ConvE_output_channels,
                 ConvE_height,
                 ConvE_width,
                 ConvE_kernel_height,
                 ConvE_kernel_width,
                 conv_e_input_dropout,
                 conv_e_output_dropout,
                 conv_e_feature_map_dropout,
                 random_seed: Optional[int] = None,
                 preferred_device: str = 'cpu'
                 ) -> None:
        super(ConvE, self).__init__(num_relations=num_relations,
                                    num_entities=num_entities)
        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device(preferred_device)
        else:
            self.device = torch.device('cpu')

        self.random_seed = random_seed
        self.stop_training = False

        self.metrics = None
        self.optimizer = None
        self.callbacks = None

        self.is_compiled = False

        # Random seeds have to set before the embeddings are initialized
        if self.random_seed is not None:
            np.random.seed(seed=self.random_seed)
            torch.manual_seed(seed=self.random_seed)
            random.seed(self.random_seed)

        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = num_entities
        #: The number of unique relation types in the knowledge graph
        self.num_relations = num_relations
        #: The dimension of the embeddings to generate
        self.embedding_dim = embedding_dim

        # Instance attributes that are defined when calling other functions
        # Calling data load function
        self.entity_label_to_id = None
        self.relation_label_to_id = None

        self.entity_embeddings = None
        self.relation_embeddings = None

        self.ConvE_height = ConvE_height
        self.ConvE_width = ConvE_width

        assert self.ConvE_height * self.ConvE_width == self.embedding_dim

        self.inp_drop = torch.nn.Dropout(conv_e_input_dropout)
        self.hidden_drop = torch.nn.Dropout(conv_e_output_dropout)
        self.feature_map_drop = torch.nn.Dropout2d(conv_e_feature_map_dropout)
        self.loss = torch.nn.BCEWithLogitsLoss()

        self.conv1 = torch.nn.Conv2d(
            in_channels=ConvE_input_channels,
            out_channels=ConvE_output_channels,
            kernel_size=(ConvE_kernel_height, ConvE_kernel_width),
            stride=1,
            padding=0,
            bias=True,
        )

        # num_features – C from an expected input of size (N,C,L)
        self.bn0 = torch.nn.BatchNorm2d(ConvE_input_channels)
        # num_features – C from an expected input of size (N,C,H,W)
        self.bn1 = torch.nn.BatchNorm2d(ConvE_output_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)
        num_in_features = ConvE_output_channels
        num_in_features *= 2 * self.ConvE_height - ConvE_kernel_height + 1
        num_in_features *= self.ConvE_width - ConvE_kernel_width + 1
        self.fc = torch.nn.Linear(num_in_features, self.embedding_dim)

        # Default optimizer for ConvE
        self.default_optimizer = torch.optim.Adam

        # Attribute for inverse models that model additional reverse
        # left side prediction embeddings.
        self.inverse_model = False

        self.label_smoothing = None

    def _init_embeddings(self):
        self.entity_embeddings = torch.nn.Embedding(self.num_entities,
                                                    self.embedding_dim)

        self.relation_embeddings = torch.nn.Embedding(self.num_relations,
                                                      self.embedding_dim)
        init_entity_biases = torch.zeros(self.num_entities)
        self.register_parameter('b', torch.nn.Parameter(init_entity_biases))
        torch.nn.init.xavier_normal_(self.entity_embeddings.weight.data)
        torch.nn.init.xavier_normal_(self.relation_embeddings.weight.data)

    def compile(self,
                metrics: Optional[List[Metric]] = None,
                callbacks: Optional[List[Callback]] = None,
                optimizer: Optional[torch.optim.Optimizer] = None,
                ) -> None:
        """
        This method has to be called after initializing the model. Otherwise,
        an error will be raised.

        The results of the metrics defined here will be returned as a training
        result. Additionally the metrics will be calculated after each epoch.

        The callbacks can be used to define additional features before and
        after each epoch.

        Parameters
        ----------
        :param metrics: Optional[List[Metric]]
            A list of metrics to calculate after each epoch.
        :param callbacks: Optional[List[Callback]]
            A list of callbacks to include during training.
        :param optimizer: torch.optim.Optimizer
            The optimizer use during training.

        Returns
        -------
        :return: None
        """
        self.optimizer = optimizer or torch.optim.Adam
        self.metrics = metrics or []
        self.callbacks = callbacks or []

        self.is_compiled = True

    def to_multi_hot(self, labels):
        batch_size = labels.shape[0]

        labels_full = torch.zeros((batch_size, self.num_entities),
                                  device=self.device)

        for i in range(batch_size):
            labels_full[i, labels[i]] = 1

        if self.label_smoothing is not None:
            labels_full = labels_full * (1.0 - self.label_smoothing)
            labels_full += (1.0 / labels_full.shape[1])

        return labels_full

    def forward(self, *inputs):
        assert len(inputs) == 1

        batch = inputs[0]

        heads = batch[:, 0:1]
        relations = batch[:, 1:2]

        x = self.forward_conv(heads, relations)

        x = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))

        x += self.b.expand_as(x)

        return x

    def forward_conv(self, subject, relation):
        batch_size = subject.shape[0]
        subj_embedded = self.entity_embeddings(subject)
        subj_embedded = subj_embedded.view(-1, 1,
                                           self.ConvE_height,
                                           self.ConvE_width)

        rel_embedded = self.relation_embeddings(relation)
        rel_embedded = rel_embedded.view(-1, 1,
                                         self.ConvE_height,
                                         self.ConvE_width)

        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = torch.cat([subj_embedded, rel_embedded], 2)

        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = self.bn0(stacked_inputs)

        # batch_size, num_input_channels, 2*height, width
        x = self.inp_drop(stacked_inputs)
        # (N,C_out,H_out,W_out)
        x = self.conv1(x)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)

        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)

        return x

    def predict_for_ranking(self, entity, relation):
        if len(entity.shape) == 0:
            entity = entity.view(-1)

        if len(relation.shape) == 1:
            relation = relation.view(-1)

        x = self.forward_conv(entity, relation)
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))
        x = x.flatten()

        x += self.b.expand_as(x)

        return x

    def score_subjects(self, p: torch.tensor, o: torch.tensor) -> torch.tensor:
        """
        Compute subject scores for all entities, given p and o.

        :param p: torch.tensor, scalar
            The predicate.
        :param o: torch.tensor, scalar
            The object.

        :return: torch.tensor, shape: (num_entities,)
            The scores for all entities.
        """
        new_p = p.clone()
        if self.inverse_model:
            new_p += self.num_relations // 2

        return self.predict_for_ranking(o, new_p)

    def score_objects(self, s: torch.tensor, p: torch.tensor) -> torch.tensor:
        """
        Compute object scores for all entities, given s and p.

        :param s: torch.tensor, scalar
            The subject.
        :param p: torch.tensor, scalar
            The predicate.

        :return: torch.tensor, shape: (num_entities,)
            The scores for all entities.
        """
        return self.predict_for_ranking(s, p)

    def filter_false_negatives(self,
                               pos_triple: Union[torch.Tensor, np.ndarray],
                               all_pos_triples: Union[torch.Tensor, np.ndarray], # noqa
                               all_entities: torch.Tensor
                               ) -> Union[torch.Tensor, torch.Tensor]:
        """
        Masks true positive and false negatives entity relation pairs.

        ..deprecated:: 0.0.1
          This function doesn't make it explicit whether the subj_mask or
          the obj_mask is returned fist. This is a potential source of bugs.
          Use filter_fn_dict() instead.
        """
        warnings.warn('Use filter_fn_dict() instead.', DeprecationWarning)
        masks = self.filter_fn_dict(pos_triple, all_pos_triples, all_entities)
        return masks['subj_mask'], masks['obj_mask']

    def filter_fn_dict(self,
                       pos_triple: Union[torch.Tensor, np.ndarray],
                       all_pos_triples: Union[torch.Tensor, np.ndarray],
                       all_entities: torch.Tensor
                       ) -> Dict[str, torch.Tensor]:
        """
        Masks true positive and false negatives entity relation pairs.

        The ConvE model performs 1:n scoring. False negatives have to be
        filtered as they would skew the rankings.

        Parameters
        ----------
        :param pos_triple: torch.Tensor or numpy.ndarray
            One positive triple from the test set.
        :param all_pos_triples: torch.Tensor or numpy.ndarray
            Will be used to determine false negative entities
        :param all_entities: torch.Tensor
            A tensor containing all possible entities.

        Returns
        -------
        :return: Dict[str: torch.Tensor]
            A dictionary containing the subject and object masks.
                'subj_mask': torch.Tensor - shape(1, num_entities)
                'obj_mask': torch.Tensor - shape(1, num_entities)
        """
        subject = pos_triple[0:1]
        relation = pos_triple[1:2]
        obj = pos_triple[2:3]

        # Create a batch where the true positives
        # are masked out
        subj_mask = all_entities != subject
        obj_mask = all_entities != obj

        # Filter corrupted triples
        true_subject_mask = all_pos_triples[:, 0:1] == subject
        true_relation_mask = all_pos_triples[:, 1:2] == relation
        true_object_mask = all_pos_triples[:, 2:3] == obj

        # Identify all samples in the training and test set that have
        # the same subject and relation as the current positive triple,
        # ie. right-side prediction (s, r, ?)
        filter_mask = (true_subject_mask & true_relation_mask)

        # Identify the corresponding object entities that correspond to
        # the subject - relation pair, i.e these are the true positive
        # object entities that will be filtered from the negatives
        objects_in_triples = all_pos_triples[:, 2:3][filter_mask]
        obj_mask[objects_in_triples] = False

        # Identify all samples in the training and test set that have
        # the same relation and object as the current positive triple,
        # i.e left-side prediction (?, r, o)
        filter_mask = (true_object_mask & true_relation_mask)

        # Identify the corresponding subject entities that correspond to
        # the relation - object pair, i.e these are the true positive
        # subject entities that will be filtered from the negatives
        subjects_in_triples = all_pos_triples[:, 0:1][filter_mask]
        subj_mask[subjects_in_triples] = False

        return dict(subj_mask=subj_mask, obj_mask=obj_mask)

    def evaluate(self,
                 test_triples: np.ndarray,
                 train_triples: np.ndarray,
                 filter_fn: Optional[str] = True,
                 use_tqdm: Optional[bool] = True
                 ) -> Dict[str, Any]:
        start = timeit.default_timer()

        ranks = []
        left_prediction_ranks = []
        right_prediction_ranks = []

        self.eval()
        self.to(self.device)

        all_pos_triples = np.concatenate([train_triples, test_triples], axis=0)
        all_pos_triples = torch.tensor(all_pos_triples, device=self.device)
        all_entities = torch.arange(self.num_entities, device=self.device)

        test_triples = torch.tensor(test_triples,
                                    dtype=torch.long,
                                    device=self.device)

        test_triples = test_triples[(test_triples[:, 1:2].flatten()).argsort()]

        if use_tqdm:
            test_triples = tqdm.tqdm(test_triples, desc=f'Corrupting triples')

        for pos_triple in test_triples:
            subj = pos_triple[0:1]
            rel = pos_triple[1:2]
            obj = pos_triple[2:3]

            subj_mask, obj_mask = self.filter_false_negatives(pos_triple,
                                                               all_pos_triples,
                                                               all_entities)

            left_prediction_scores = self.score_subjects(o=obj, p=rel)
            right_prediction_scores = self.score_objects(s=subj, p=rel)

            true_subj_score = left_prediction_scores[subj]
            true_obj_score = right_prediction_scores[obj]

            if filter_fn:
                left_prediction_scores = left_prediction_scores[subj_mask]
                right_prediction_scores = right_prediction_scores[obj_mask]

            rank_left_prediction = left_prediction_scores > true_subj_score
            rank_left_prediction = rank_left_prediction.sum() + 1
            rank_left_prediction = rank_left_prediction.item()

            rank_right_prediction = right_prediction_scores > true_obj_score
            rank_right_prediction = rank_right_prediction.sum() + 1
            rank_right_prediction = rank_right_prediction.item()

            ranks.append(rank_left_prediction)
            ranks.append(rank_right_prediction)
            right_prediction_ranks.append(rank_right_prediction)
            left_prediction_ranks.append(rank_left_prediction)

        result = dict()
        for metric in self.metrics:
            parameters = dict()
            parameters['subj_ranks'] = left_prediction_ranks
            parameters['obj_ranks'] = right_prediction_ranks
            parameters['ranks'] = ranks

            result[metric.name] = metric(parameters)

        stop = timeit.default_timer()
        logger.info("Evaluation took %.2f seconds", stop-start)

        return result

    def fit(self,
            train_triples: np.ndarray,
            val_triples: np.ndarray,
            num_epochs: int,
            learning_rate: int,
            batch_size: int,
            label_smoothing: Optional[bool] = False,
            train_reversed: Optional[bool] = True,
            weight_decay: Optional[float] = 0,
            validation_interval: Optional[int] = 1
            ) -> Tuple[List[float], Dict[str, List[int]]]:
        start = timeit.default_timer()
        logger.info('Creating inverse triples')

        if val_triples is not None:
            val_metrics = dict()

        if train_reversed:
            logger.info('Create reversed triples to predict left side (?, r, o)') # noqa

            train_triples = self._create_reversed_triples(train_triples)

            # When the model has not been trained as inverse_model before,
            # the number of relations have to be doubled
            if not self.inverse_model:
                self.num_relations *= 2
                self.inverse_model = True

        self._init_embeddings()

        if label_smoothing is not None:
            self.label_smoothing = label_smoothing

        logger.info('Created inverse triples. It took %.2f seconds', timeit.default_timer() - start) # noqa
        logger.info('Grouping entity relation pairs')
        group = self._group_enitity_relation(train_triples)
        labels_list = np.array(list(group.values()))
        unique_s_p = np.array(list(group.keys()))

        # Now we can overwrite train_triples
        trainX = unique_s_p

        self.to(self.device)

        # Initialize the optimizer given as attribute
        self.optimizer = self.optimizer(self.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay)

        logger.info('****Run Model On %s ****', str(self.device).upper())

        loss_per_epoch = []
        num_pos_triples = trainX.shape[0]

        start_training = timeit.default_timer()

        epoch_loss = None

        _tqdm_kwargs = dict(desc='Training epoch')

        indices = np.arange(num_pos_triples)
        trange_bar = tqdm.trange(num_epochs, **_tqdm_kwargs)

        for epoch in trange_bar:
            self.train()
            np.random.shuffle(indices)
            trainX = trainX[indices]
            labels_list = np.array([labels_list[i] for i in indices])

            pos_batches = split_list_in_batches(input_list=trainX,
                                                batch_size=batch_size)

            labels_batches = split_list_in_batches(input_list=labels_list,
                                                   batch_size=batch_size)

            current_epoch_loss = self._fit_batches(pos_batches, labels_batches)

            # Track epoch loss
            previous_loss = epoch_loss
            epoch_loss = current_epoch_loss
            epoch_loss /= len(trainX) * self.entity_embeddings.num_embeddings # noqa

            loss_per_epoch.append(epoch_loss)

            trange_bar.set_postfix(loss=epoch_loss,
                                   previous_loss=previous_loss)

            if val_triples is not None and epoch % validation_interval == 0:
                with torch.no_grad():
                    metrics = self.evaluate(val_triples,
                                            train_triples)

                for metric_key, metric_val in metrics.items():
                    entry = val_metrics.get(f'val_{metric_key}', [])
                    entry.append(metric_val)

                    val_metrics[f'val_{metric_key}'] = entry

            callback_param = dict()
            for c in self.callbacks:
                callback_param['losses'] = loss_per_epoch
                callback_param['val_metrics'] = val_metrics
                callback_param['model'] = self
                c.on_epoch_end(epoch, callback_param)

            if self.stop_training:
                logger.info('Training stopped at epoch %d', epoch)
                break

        stop_training = timeit.default_timer()
        logger.info("Training took %.2f seconds \n", str(round(stop_training - start_training))) # noqa

        return loss_per_epoch, val_metrics

    def _fit_batches(self, pos_batches, labels_batches):
        current_epoch_loss = 0.
        training_data = zip(pos_batches, labels_batches)

        for _, (pos_batch, labels_batch) in enumerate(training_data):
            current_batch_size = len(pos_batch)

            pos_batch = torch.tensor(pos_batch,
                                     dtype=torch.long,
                                     device=self.device)

            self.optimizer.zero_grad()

            scores = self(pos_batch)
            labels_full = self.to_multi_hot(labels_batch)
            loss = self.loss(scores, labels_full)

            scaled_loss = loss.item() * current_batch_size
            scaled_loss *= self.entity_embeddings.num_embeddings
            current_epoch_loss += scaled_loss

            loss.backward()
            self.optimizer.step()

        return current_epoch_loss

    def _create_reversed_triples(self, triples):
        logger.debug('Crate Reverse Triples')
        inverse_triples = np.flip(triples.copy())
        inverse_triples[:, 1:2] += self.num_relations
        triples = np.concatenate((triples, inverse_triples))

        return triples

    def _group_enitity_relation(self, train_triples):
        from collections import defaultdict
        triple_dict = defaultdict(set)
        for row in tqdm.tqdm(train_triples):
            triple_dict[(row[0], row[1])].add(row[2])

        # Create lists out of sets for numpy indexing when loading the labels
        triple_dict = {key: list(value) for key, value in triple_dict.items()}
        return triple_dict


class ConveEvaluationModel(EvaluationModel):
    def __init__(self,
                 init_model: ConvE,
                 all_positive_triples: torch.tensor,
                 all_entities: torch.tensor,
                 filter_negatives: bool = True,
                 device: str = 'cpu'):
        """
        Initialzes a ConvE evaluation model.

        Parameters
        ----------
        :params init_model: ConvE
            A trained ConvE model.
        :params all_positive_triples: torch.tensor
            All positive triples including the training and test
            triples.
        :params all_entities: torch.tensor
            All entities in the dataset.
        :params filter_negatives: bool
            If true (default) filter false negative triples.
        """
        super(ConveEvaluationModel, self).__init__(None, None)
        self.filter_negatives = filter_negatives
        self.all_positive_triples = all_positive_triples
        self.all_entities = all_entities
        self.model = init_model
        self.model.eval()

        self.cache = dict()

        self.device = torch.device(device) or self.model.device

    def get_filter_masks(self, triple):
        """
        Attempts to retrieve the cached mask if available otherwise
        will recompute it.
        """
        if triple not in self.cache:
            masks = self.model.filter_fn_dict(
                triple, self.all_positive_triples, self.all_entities
            )
            self.cache[triple] = (masks['obj_mask'], masks['subj_mask'])

        return self.cache[triple]

    def predict_object_scores(self, s: torch.tensor, p: torch.tensor,
                              o: torch.tensor) -> torch.tensor:
        batch_size = s.shape[0]

        out_scores = []
        for i in range(batch_size):
            with torch.no_grad():
                scores = self.model.score_objects(s[i],
                                                  p[i])

            if self.filter_negatives:
                triple = [s[i], p[i], o[i]]
                triple = torch.tensor(triple,
                                      dtype=torch.long,
                                      device=self.device)

                obj_mask, _ = self.get_filter_masks(triple)
                obj_mask[o[i]] = True
                scores[obj_mask == 0] = scores.min()

            out_scores.append(scores.view(1, -1))

        return torch.cat(out_scores, dim=0)

    def predict_subject_scores(self, s: torch.tensor, p: torch.tensor,
                               o: torch.tensor) -> torch.tensor:
        batch_size = s.shape[0]

        out_scores = []
        for i in range(batch_size):
            with torch.no_grad():
                scores = self.model.score_subjects(p[i],
                                                   o[i])

            if self.filter_negatives:
                triple = [s[i], p[i], o[i]]
                triple = torch.tensor(triple,
                                      dtype=torch.long,
                                      device=self.device)

                _, subj_mask = self.get_filter_masks(triple)
                subj_mask[s[i]] = True

                scores[subj_mask == 0] = scores.min()

            out_scores.append(scores.view(1, -1))

        return torch.cat(out_scores, dim=0)
