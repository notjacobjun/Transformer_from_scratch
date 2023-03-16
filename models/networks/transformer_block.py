import torch
from torch import nn

from models.layers.self_attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout_prob, forward_expansion=4) -> None:
        super().__init__()

        # setup the network architecture
        self.attention = SelfAttention(embed_size, heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout_prob)

    # TODO consider adding the masking capability for increased performance
    def forward(self, X, mask):
        # apply self attention to the input
        attended = self.attention(X, mask)

        # apply the residual connections and normalization layer
        normalized_weights = self.dropout(self.norm1(attended + X))

        # apply the FFN layer
        fedforward_weights = self.ff(normalized_weights)

        # apply the normalization
        return self.dropout(self.norm2(fedforward_weights + X))
