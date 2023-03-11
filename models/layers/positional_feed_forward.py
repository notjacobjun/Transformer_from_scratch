from torch import nn

class PositionalFeedForward(nn.Module):
    def __init__(self, num_heads, num_FFN, n_layers, encoder_voc_size, d_model) -> None:
        self.encoder_voc_size = encoder_voc_size
        