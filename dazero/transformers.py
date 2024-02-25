import numpy as np

import dazero.functions as F
from dazero.layers import Layer, Linear


class SelfAttention(Layer):
    def __init__(self, dim, heads=4, mask=False):
        super().__init__()

        if dim % heads != 0:
            raise RuntimeError("input vector size must be an integer multiple of heads")
        
        self.dim, self.heads = dim, heads

        # Compute the queries, keys and values for all heads
        self.toKeys = Linear(dim, dim, bias=False)
        self.toQueries = Linear(dim, dim, bias=False)
        self.toValues = Linear(dim, dim, bias=False)

        # This will be applied after the multi-head self-attention operation.
        self.unifyHeads = Linear(dim, dim)
    
    def forward(self, x):
        b, t, k = x.shape
        h = self.heads

        queries = self.toQueries(x)
        keys = self.toKeys(x)
        values = self.toValues(x)

        s = k // h
        # queries/keys/values are cut into pieces (dimensionality reduction) and sent to different heads respectively
        queries = F.reshape(queries, shape=(b, t, h, s))
        keys = F.reshape(keys, shape=(b, t, h, s))
        values = F.reshape(values, shape=(b, t, h, s))

        # fold heads into the batch dimension
        queries = queries.transpose(0, 2, 1, 3).reshape(b * h, t, s)
        keys = keys.transpose(0, 2, 1, 3).reshape(b * h, t, s)
        values = values.transpose(0, 2, 1, 3).reshape(b * h, t, s)

        # Get dot product of queries and keys, and scale
        dot = F.bmm(queries, keys.transpose(0, 2, 1))
        dot = dot / (k ** (1/2))
        dot = F.softmax(dot, axis=2)

        # apply the self attention to the values
        out = F.bmm(dot, values).reshape(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(0, 2, 1, 3).reshape(b, t, s * h)
        return self.unifyHeads(out)
