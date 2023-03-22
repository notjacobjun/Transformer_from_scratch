import torch
from torch import nn
from models.embedding.positional_embedding import PositionalEmbedding

from models.embedding.token_embedding import TokenEmbedding
from models.layers.transformer_block import TransformerBlock


class Encoder(nn.Module):
    def __init__(self, heads, max_len, device, embed_size, n_layers, dropout_prob, vocab_size, forward_expansion):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.device = device

        self.word_embedding = TokenEmbedding(vocab_size, embed_size)
        self.positional_embedding = PositionalEmbedding(max_len, embed_size)
        # TODO setup positional encoding (need to fix the dimensional difference between feeding mechanism when using encoding)
        # self.positional_encoding = PositionalEncoding(
        #     embed_size, max_len, device)

        self.layers = nn.ModuleList([TransformerBlock(
            embed_size, heads, dropout_prob, forward_expansion) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X, mask):
        N, seq_length = X.size()
        positions = torch.arange(
            0, seq_length, device=self.device).expand(N, seq_length)

        out = self.dropout(self.word_embedding(
            X) + self.positional_embedding(positions))

        for layer in self.layers:
            out = layer(X, mask)

        return out
