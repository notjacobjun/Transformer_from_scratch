import torch
from torch import nn

from models.layers.self_attention import SelfAttention
from models.layers.transformer_block import TransformerBlock


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout_prob, device) -> None:
        super(DecoderBlock, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout_prob, forward_expansion)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X, src_mask, trg_mask):
        attention = self.attention(X, trg_mask)

        out = self.dropout(self.norm1(attention + X))
        out = self.transformer_block(X, src_mask)
        return out
