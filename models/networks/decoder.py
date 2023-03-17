import torch
from torch import nn
from models.embedding.positional_encoding import PositionalEncoding

from models.embedding.token_embedding import TokenEmbedding
from models.layers.decoder_block import DecoderBlock
from models.layers.transformer_block import TransformerBlock


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, n_layers, heads, forward_expansion, dropout_prob, device, max_len) -> None:
        super(Decoder, self).__init__()
        self.device = device

        self.word_embedding = TokenEmbedding(trg_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(
            embed_size, max_len, device)
        self.layers = nn.ModuleList([DecoderBlock(
            embed_size, heads, forward_expansion, dropout_prob, device) for _ in range(n_layers)])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X, enc_out, src_mask, trg_mask):
        # parse the shape from the input matrix
        N, seq_len = X.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        X = self.dropout(self.word_embedding(X) + self.positional_encoding(X))

        for layer in self.layers:
            X = layer(X, src_mask, trg_mask)

        out = self.fc_out(X)

        return out
