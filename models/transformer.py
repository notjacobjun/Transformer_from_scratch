from torch import nn


class Transformer(nn.Module):
    # TODO modify constructor to take in the hyperparameters needed for sub modules of transformer
    def __init__(self, device) -> None:
        super(Transformer, self).__init__()
        self.device = device

        # TODO setup the high level structure of the encoder and decoder
        self.encoder = None

        self.decoder = None

        # TODO figure out the logic for trg_vocab_size, src_pad_idx, and other stuff
        # TODO create src mask and trg mask helper functions
