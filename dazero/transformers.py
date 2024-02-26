import dazero
from dazero import cuda
import dazero.functions as F
from dazero.layers import Linear, LayerNorm, Embedding


def mask_(matrices, val=0.0):
    h, w = matrices.shape[-2], matrices.shape[-1]
    assert h == w
    xp = cuda.get_array_module(matrices)
    for i in range(matrices.shape[0]):
        tril_indices = xp.triu_indices(h, k=1)
        matrices.data[i][tril_indices] = val


class SelfAttention(dazero.Layer):
    def __init__(self, dim, heads=4, mask=False):
        super().__init__()

        if dim % heads != 0:
            raise RuntimeError("input vector size must be an integer multiple of heads")

        self.dim = dim
        self.heads = heads
        self.mask = mask

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

        # mask out the upper half of the dot matrix, excluding the diagonal
        if self.mask:
            mask_(dot, val=float('-inf'))

        dot = F.softmax(dot, axis=2)

        # apply the self attention to the values
        out = F.bmm(dot, values).reshape(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(0, 2, 1, 3).reshape(b, t, s * h)
        return self.unifyHeads(out)


class TransformerBlock(dazero.Layer):
    def __init__(self, dim, heads, mask, ffn_hidden_mult=4, dropout=0.0):
        super().__init__()
        self.dropout = dropout

        self.attention = SelfAttention(dim, heads=heads, mask=mask)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.ffn = dazero.models.Sequential(
            Linear(dim, ffn_hidden_mult * dim),
            F.ReLU(),
            Linear(ffn_hidden_mult * dim, dim)
        )
    
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ffn(x)
        x = self.norm2(fedforward + x)
        x = F.dropout(x, dropout_ratio=self.dropout)
        return x


class Transformer(dazero.Model):
    def __init__(self, dim, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = Embedding(num_tokens, dim)
        self.pos_emb = Embedding(seq_length, dim)

        # The sequence of transformer blocks
        tblocks = []
        for _ in range(depth):
            tblocks.append(TransformerBlock(dim=dim, heads=heads))
        self.tblocks = dazero.models.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = Linear(dim, num_classes)

    def forward(self, x):
        # generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.shape

        # generate position embeddings
        xp = cuda.get_array_module(x)
        positions = xp.arange(t)
        positions = self.pos_emb(positions)
        positions = F.broadcast_to(positions, (b, t, k))

        x = tokens + positions
        x = self.tblocks(x)

        # Average-pool over the t dimension and project to class probabilities
        x = self.toprobs(x.mean(axis=1))
        x = F.log_softmax(x, axis=1)
