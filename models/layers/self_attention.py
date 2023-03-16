import torch
import math
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads=4, mask=False):
        super.__init__()

        # Ensure that k evenly divides the num_heads
        assert embed_size % num_heads == 0

        # set the attributes
        self.k, self.mask, self.num_heads = embed_size, mask, num_heads

        # parse the input into keys, query, and values
        self.toqueries = nn.Linear(embed_size, embed_size, bias=False)
        self.tokeys = nn.Linear(embed_size, embed_size, bias=False)
        self.tovalues = nn.Linear(embed_size, embed_size, bias=False)

        # this layer will be applied after the multi-head attention operation is performed
        self.unifyheads = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # parse the size
        b, t, k = x.size()
        h = self.num_heads

        # parse the queries, keys, and values from the input x
        queries = self.toqueries(x)
        keys = self.tokeys(x)
        values = self.tovalues(x)

        # split the keys, queries, adn values into s chunks of matrix operations (this is for efficient multi-head attention)
        s = k // h

        # reshaping the tensors
        keys = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # transpose the matrices to reshape them into 3D tensors, therefore allowing us to use the torch.bmm function for
        # more efficient computation (we are transposing to reduce the head dimension since all the dot products will be same)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # compute the dot product of all the queries and keys using batch operations
        weights = torch.bmm(keys, queries)

        # scale the weights by dividing by sqrt(d_model) for problem of vanishing gradients
        weights /= (math.sqrt(k))

        # optionally we can mask out leftward information flow to preserve auto-regressive property
        if self.mask:
            # TODO perform the masking logic here
            pass

        # perform softmax on weights (rescale weights to sum to 1)
        nn.functional.softmax(weights, dim=2)

        # find the dot product of these weights with the values then reshape to original dimensions
        weights = torch.bmm(weights, values).view(b, h, t, s)

        # return the unifications of these heads
        return self.unifyheads(weights)
