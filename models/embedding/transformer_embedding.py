from torch import nn
from models.embedding.positional_encoding import PositionalEncoding

from models.embedding.token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    def __init__(self, seq_length, dimen_model, drop_prob) -> None:
        """
        params:
        seq_length: The length of the current sequence
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, dimen_model)
        self.positional_encoding = PositionalEncoding(seq_length, dimen_model)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self):
        pass
