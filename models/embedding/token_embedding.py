from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Using torch.nn Embedding to convert the input (text) into dense representation of weighted matrix
    """

    def __init__(self, vocab_size, embed_size) -> None:
        super(TokenEmbedding).__init__(vocab_size, embed_size, padding_idx=1)
