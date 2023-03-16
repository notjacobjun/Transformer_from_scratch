from torch import nn
from models.embedding.positional_encoding import PositionalEncoding

from models.embedding.token_embedding import TokenEmbedding
from models.networks.transformer import Transformer


class Encoder(nn.Module):
    def __init__(self, heads, max_len, device, embed_size, forward_expansion, ffn_hidden, n_layers, dropout_prob, vocab_size):
        super(Encoder, self).__init__()
        self.word_embedding = TokenEmbedding(vocab_size, embed_size)
        self.positional_embedding = PositionalEncoding(max_len, embed_size)

    def forward(self):
        pass
