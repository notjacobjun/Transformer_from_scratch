import torch


class PositionalEncoding:
    def __init__(self, embed_size, max_len, device, n=10000) -> None:
        self.embed_size = embed_size
        self.max_len = max_len
        self.device = device
        self.n = n

        # convert the position vector into 2D to represent the word's position
        self.position = torch.arange(
            self.max_len, device=self.device).unsqueeze(1).float()

        # since we don't need to compute the gradient we can make it faster by detaching
        self.positional_encoding = torch.zeros(
            1, self.max_len, self.embed_size, device=self.device).detach()

        # This is the exponent that is used for the sine and cosine positional encoding formula
        _2i = torch.arange(0, self.embed_size, step=2,
                           device=self.device).float()

        print(f"position shape: {self.position.shape}")
        print(
            f"positional encoding shape: {self.positional_encoding[0, 1:2].shape}")
        # PE(pos, 2i) = sin(pos / n^(2i/embed_size))
        self.positional_encoding[0, 0::2] = torch.sin(
            self.position / (n ** (_2i / self.embed_size)))

        # PE(pos, 2i+1) = cos(pos / n^(2i/embed_size))
        self.positional_encoding[0, 1:2] = torch.cos(
            self.position / (n ** (_2i / self.embed_size)))

    def forward(self, X):
        # (batch_size, input_len, embed_size)
        batch_size, seq_length = X.size()

        return self.positional_encoding[:seq_length, :]
