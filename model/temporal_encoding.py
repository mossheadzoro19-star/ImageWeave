import torch
import torch.nn as nn


class TemporalEncoding(nn.Module):

    def __init__(self, hidden_dim=768, max_images=3):

        super().__init__()

        self.embedding = nn.Embedding(
            max_images,
            hidden_dim
        )

    def forward(self, x):

        B, T, N, D = x.shape

        positions = torch.arange(
            T,
            device=x.device
        )

        temporal_embeddings = self.embedding(positions)

        temporal_embeddings = temporal_embeddings.unsqueeze(1)

        x = x + temporal_embeddings

        return x