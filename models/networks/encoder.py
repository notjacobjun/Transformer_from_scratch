import torch
from torch import nn
from models.embedding.positional_encoding import PositionalEncoding

from models.embedding.token_embedding import TokenEmbedding
from models.networks.transformer_block import TransformerBlock


class Encoder(nn.Module):
    def __init__(self, heads, max_len, device, embed_size, n_layers, dropout_prob, vocab_size, forward_expansion=4):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.device = device

        self.word_embedding = TokenEmbedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(
            embed_size, max_len, device)

        self.layers = nn.ModuleList([TransformerBlock(
            embed_size, heads, dropout_prob, forward_expansion) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X, mask):
        N, seq_length = X.shape()
        # TODO check if we need positions array when using positional encoding instead of pos embedding
        positions = torch.arange(
            0, seq_length, device=self.device).expand(N, seq_length)

        # TODO check if we are supposed to pass in positions array rather than X (b/c embedding method uses positions array but not sure if encoding also uses it)
        out = self.dropout(self.word_embedding(
            X) + self.positional_encoding(X))

        for layer in self.layers:
            out = layer(X, mask)

        return out
