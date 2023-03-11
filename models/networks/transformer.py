import torch
from torch import nn

from models.layers.self_attention import SelfAttention


class Transformer(nn.Module):
    def __init__(self, k, heads) -> None:
        super().__init__()

        # setup the network architecture
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, x):
        # apply self attention to the input
        attended = self.attention(x)

        # apply the residual connections and normalization layer
        normalized_weights = self.norm1(attended + x)

        # apply the FFN layer
        fedforward_weights = self.ff(normalized_weights)

        # apply the normalization
        return self.norm2(fedforward_weights + x)
