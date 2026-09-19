import torch
import torch.nn as nn

class DeepSetsRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(DeepSetsRegressor, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Regressor (Mean + Max + Std) -> 3 * hidden_dim
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [Batch, N, input_dim]

        features = self.encoder(x) # -> [Batch, N, Hidden]

        # Aggregate
        mean_pool = torch.mean(features, dim=1)
        max_pool = torch.max(features, dim=1)[0]
        std_pool = torch.std(features, dim=1, unbiased=False)

        global_feature = torch.cat([mean_pool, max_pool, std_pool], dim=1)
        #global_feature = torch.cat([std_pool], dim=1)

        mass = self.regressor(global_feature)
        return mass