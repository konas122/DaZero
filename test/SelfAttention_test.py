import os
try:
    import torch
except ImportError:
    os._exit(1)
import numpy as np
from torch import nn
import torch.nn.functional as F

from dazero import transformers, Parameter


class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False):
        super().__init__()

        assert k % heads == 0 # input vector size 必须是 heads 的整数倍
        self.k, self.heads = k, heads
        # Compute the queries, keys and values for all heads
        self.tokeys    = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues  = nn.Linear(k, k, bias=False)

        # This will be applied after the multi-head self-attention operation.
        self.unifyheads = nn.Linear(k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        # 首先，为所有 heads 计算 query/key/value，得到的是完整嵌入维度的 k*k 矩阵
        queries = self.toqueries(x)
        keys    = self.tokeys(x)
        values  = self.tovalues(x)

        # 接下来将 queries/keys/values 切块（降维），分别送到不同的 head
        s = k // h
        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # Get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2)) # -- dot has size (b*h, t, t) containing raw weights
        dot = dot / (k ** (1/2))                       # scale the dot product
        dot = F.softmax(dot, dim=2)                    # normalize, dot now contains row-wise normalized weights
        out = torch.bmm(dot, values).view(b, h, t, s) # apply the self attention to the values
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)


x = np.random.randn(10, 3, 4).astype(np.float32)
x_torch = torch.from_numpy(x)
model_torch = SelfAttention(4)

model = transformers.SelfAttention(4)

model.toKeys.W.data = model_torch.tokeys.weight.T.detach().numpy()
model.toQueries.W.data = model_torch.toqueries.weight.T.detach().numpy()
model.toValues.W.data = model_torch.tovalues.weight.T.detach().numpy()
model.unifyHeads.W.data = model_torch.unifyheads.weight.T.detach().numpy()
model.unifyHeads.b.data = model_torch.unifyheads.bias.detach().numpy()

x = Parameter(x)
assert np.allclose(model(x).data, model_torch(x_torch).detach().numpy(), atol=1e-7) == True
print("Success")
