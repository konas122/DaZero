import os

try:
    import torch
except ImportError:
    os._exit(0)

import numpy as np
import dazero.functions as D
import torch.nn.functional as F
from dazero import Tensor, Variable


np.random.seed(0)
x = np.random.randn(2, 3, 4).astype(np.float32)
x_torch = torch.from_numpy(x).requires_grad_()
x = Tensor(x)

W = np.random.randn(4, 3).astype(np.float32)
W_torch = torch.from_numpy(W).requires_grad_()
W = Tensor(W)

res = D.linear(x, W, None)
res_torch = F.linear(x_torch, W_torch.T, None)

assert np.allclose(res_torch.detach().numpy(), res.data, atol=1e-7) == True

grad = np.random.randn(*res.shape)
res.grad = Variable(grad)
res.backward()
res_torch.backward(torch.from_numpy(grad))
assert np.allclose(x.grad.data, x_torch.grad.numpy(), atol=1e-7) == True

print("Success")
