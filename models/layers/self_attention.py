import torch
import math
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads=4):
        super().__init__()

        # Ensure that k evenly divides the num_heads
        assert embed_size % num_heads == 0

        # set the attributes
        self.k, self.num_heads, self.heads_dim = embed_size, num_heads, embed_size // num_heads

        # parse the input into keys, query, and values
        self.toqueries = nn.Linear(embed_size, embed_size, bias=False)
        self.tokeys = nn.Linear(embed_size, embed_size, bias=False)
        self.tovalues = nn.Linear(embed_size, embed_size, bias=False)

        # this layer will be applied after the multi-head attention operation is performed
        self.unifyheads = nn.Linear(embed_size, embed_size)

    def forward(self, X, mask):
        N, t = X.shape[0], X.shape[1]
        # parse the queries, keys, and values from the input x
        print(f"embed_size: {self.k}")
        print(f"N: {N} and t {t}")
        queries = self.toqueries(X)
        keys = self.tokeys(X)
        values = self.tovalues(X)

        # reshaping the tensors into smaller parts of heads for efficiency
        keys = keys.reshape(N, self.k, self.num_heads, self.heads_dim)
        queries = queries.reshape(
            N, self.k, self.num_heads, self.heads_dim)
        values = values.reshape(
            N, self.k, self.num_heads, self.heads_dim)

        # transpose the matrices to reshape them into 3D tensors, therefore allowing us to use the torch.bmm function for
        # more efficient computation (we are transposing to reduce the head dimension since all the dot products will be same)
        keys = keys.transpose(1, 2).contiguous().view(
            N * self.num_heads, t, self.heads_dim)
        queries = queries.transpose(1, 2).contiguous().view(
            N * self.num_heads, t, self.heads_dim)
        values = values.transpose(1, 2).contiguous().view(
            N * self.num_heads, t, self.heads_dim)

        # compute the dot product of all the queries and keys using batch operations
        weights = torch.bmm(keys, queries)

        # scale the weights by dividing by sqrt(d_model) for problem of vanishing gradients
        weights /= (math.sqrt(self.k))

        # optionally we can mask out leftward information flow to preserve auto-regressive property
        if mask is not None:
            # note that instead of using -infinity we are using some really small value to prevent overflow
            weights = weights.masked_fill(mask == 0, float("-1e20"))

        # perform softmax on weights (rescale weights to sum to 1)
        nn.functional.softmax(weights, dim=2)

        # find the dot product of these weights with the values then reshape to original dimensions
        weights = torch.bmm(weights, values).view(
            N, self.num_heads, t, self.heads_dim)

        # return the unifications of these heads
        return self.unifyheads(weights)
