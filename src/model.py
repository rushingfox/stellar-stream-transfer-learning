import torch
import torch.nn as nn

class DeepSetsRegressor(nn.Module):
    """DeepSets regressor: a per-particle MLP encoder, mean + max + std pooling over particles, then an MLP head with one output.

    `input_dim`: the features per particle, 3 for x, y, z.
    `hidden_dim`: the encoder width, main.py's --hidden_dim (1024 by default there).
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(DeepSetsRegressor, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # The head takes the three pooled vectors (mean, max, std) concatenated, 3 * hidden_dim.
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """[batch, n_particles, input_dim] -> [batch, 1], the predicted z-scored target."""
        features = self.encoder(x) # -> [Batch, N, Hidden]

        # Pool over the particle axis; std is the population std (unbiased=False).
        mean_pool = torch.mean(features, dim=1)
        max_pool = torch.max(features, dim=1)[0]
        std_pool = torch.std(features, dim=1, unbiased=False)

        global_feature = torch.cat([mean_pool, max_pool, std_pool], dim=1)
        #global_feature = torch.cat([std_pool], dim=1)

        mass = self.regressor(global_feature)
        return mass