import numpy as np
import torch


class PositionalEncoding:
    def __init__(self, embed_size, max_len, device, n=10000) -> None:
        self.embed_size = embed_size
        self.max_len = max_len
        self.n = n
        self.device = device

        self.position = torch.arange(
            self.max_len, device=self.device).unsqueeze(1)

        self.positional_encoding = torch.zeros(
            1, self.max_len, self.embed_size, device=self.device)

        # This is the exponent that is used for the sine and cosine positional encoding formula
        _2i = torch.arange(0, self.embed_size, step=2,
                           device=self.device).float()

        # calculate the position for each

        # loop through each corresponding index and use the sin and cos functions to get the resulting encoding
        # even indexed formula
        # PE(pos, 2i) = sin(pos / n^(2i/embed_size))
        self.positional_encoding[0, :, 0::2] = torch.sin(
            self.position / (n ** _2i / self.embed_size))

        # PE(pos, 2i+1) = cos(pos / n^(2i/embed_size))
        self.positional_encoding[0, :, 1:2] = torch.cos(
            self.position / (n ** _2i / self.embed_size))

    def forward(self, X):
        # (batch_size, input_len, embed_size)
        batch_size, seq_length, _ = X.size()

        return self.positional_encoding[:batch_size, :seq_length, :]
