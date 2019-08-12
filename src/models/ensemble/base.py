import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from tqdm import trange
from tqdm import tqdm
import numpy as np

from src.models.ensemble.utils import to_multihot
from src.metrics import HitsAtK, MeanRank, MeanReciprocalRank
from src.utils import calculate_ranks


class EnsembleModel(torch.nn.Module):
    def __init__(self, ensemble_units, num_entities,  preferred_device='cpu'):
        super(EnsembleModel, self).__init__()
        self.ensemble_units = ensemble_units
        self.preferred_device = preferred_device

        self.dense = torch.nn.Linear(len(ensemble_units), 1)

        self.device = preferred_device
        self.num_entities = num_entities

        # Fix models' parameters
        for model in self.ensemble_units:
            for param in model.parameters():
                param.requires_grad = False

        # Send everything to device
        for model in self.ensemble_units:
            model.to(self.device)

        # Logistic Regression
        self.linear = torch.nn.Linear(in_features=len(ensemble_units),
                                      out_features=1,
                                      bias=True).to(device=preferred_device)

        self.loss = nn.BCEWithLogitsLoss(reduction='sum')

        # Optimize only logistic regression weights
        self.optimizer = Adam(self.linear.parameters())

    def forward(self, *inputs):
        assert len(inputs) == 1
        batch = inputs[0]
        # Predict scores for all objects, in all models: (n_entities, n_models)
        scores = torch.stack([model.forward(batch)
                             for model in self.ensemble_units], dim=-1)

        logits = self.linear.forward(scores)

        return logits

    def evaluate(self, X, batch_size=128):
        x_tensor = torch.from_numpy(X).to(device=self.preferred_device)
        test_set = TensorDataset(x_tensor)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        all_ranks = []

        self.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):

                batch = batch[0]
                # scores shape (batch_size, num_ensemble_units, num_entities)
                all_scores = self(batch[:, 0:2]).squeeze()

                row_idx = torch.arange(batch.shape[0]).view(-1, 1)
                true_entity_scores = all_scores[row_idx, batch[:, 2].view(-1, 1)] # noqa
                ranks = calculate_ranks(true_entity_scores, all_scores)
                all_ranks.append(ranks.detach().cpu().numpy())

        parameters = dict(ranks=np.concatenate(all_ranks))

        return {
            'hits_at_1': HitsAtK(1).calculate_metric(parameters),
            'hits_at_3': HitsAtK(3).calculate_metric(parameters),
            'hits_at_10': HitsAtK(10).calculate_metric(parameters),
            'mr': MeanRank().calculate_metric(parameters),
            'mrr': MeanReciprocalRank().calculate_metric(parameters),
        }

    def fit(self, X, epochs, batch_size=128):
        x, y = to_multihot(positive_triples=X)

        n_samples = len(x)

        for _ in trange(epochs,
                             unit='epoch',
                             unit_scale=True,
                             desc='Epoch'):

            epoch_loss = 0.0

            for i in trange(0, n_samples, batch_size,
                                 unit='sample',
                                 unit_scale=True,
                                 desc='Fit Batches'):
                # Create mini-batch
                k = min(i + batch_size, n_samples)
                x_batch = x[i:k]

                # convert to one hot
                y_batch = torch.zeros((k - i, self.num_entities))
                
                for batch_idx, _ in enumerate(y_batch):
                    offset = i + batch_idx
                    true_positives = y[offset]
                    y_batch[batch_idx, true_positives] = 1.0

                y_batch = y_batch.to(device=self.preferred_device)

                # Zero gradient
                self.optimizer.zero_grad()

                # Forward pass
                x_batch = torch.from_numpy(x_batch)
                x_batch = x_batch.to(device=self.preferred_device)

                logits = self.forward(x_batch).squeeze()
                loss = self.loss(logits, y_batch)

                # Compute gradients
                loss.backward()

                # Accumulate epoch loss
                epoch_loss += loss.item()

                # Update parameters
                self.optimizer.step()
