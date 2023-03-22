from torch import nn


class PositionalEmbedding(nn.Embedding):
    """
    Using torch.nn Embedding to convert the positional information into dense representation of weighted matrix
    """

    def __init__(self, max_length, embed_size) -> None:
        """
        :param vocab_size: size of vocabulary
        :param embed_size: dimensions of model
        """
        super(PositionalEmbedding, self).__init__(
            max_length, embed_size)
