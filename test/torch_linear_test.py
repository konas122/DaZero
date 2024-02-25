import os

try:
    import torch
except ImportError:
    os._exit(0)

import numpy as np
import dazero.functions as L
import torch.nn.functional as F
from dazero import Parameter


x = np.random.randn(3, 4).astype(np.float32)
x_torch = torch.from_numpy(x)
x = Parameter(x)

W = np.random.randn(4, 3).astype(np.float32)
W_torch = torch.from_numpy(W)
W = Parameter(W)

assert np.allclose(F.linear(x_torch, W_torch.T, None).numpy(), L.linear(x, W, None).data, atol=1e-7) == True
print("Success")
