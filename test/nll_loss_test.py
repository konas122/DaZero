import os
try:
    import torch
except ImportError:
    os._exit(0)
import numpy as np
import torch.nn.functional as F

import dazero.functions as D
from dazero import Parameter


b = 2
n = 3
dim = 5

# ============================== 2D Tensor =================================

x = np.random.randn(n, dim)
x_torch = torch.from_numpy(x).requires_grad_()
x = Parameter(x)

target = np.random.randint(0, n, size=(n))
target_torch = torch.from_numpy(target)
target = Parameter(target)

y = D.nll_loss(x, target.data)
y_torch = F.nll_loss(x_torch, target_torch.long())

y.backward()
y_torch.backward()

assert np.allclose(y.data, y_torch.detach().numpy(), atol=1e-8)
assert np.allclose(x.grad.data, x_torch.grad.numpy(), atol=1e-8)


# =========================== reduction='sum' =============================

y = D.nll_loss(x, target.data, reduction='sum')
y_torch = F.nll_loss(x_torch, target_torch.long(), reduction='sum')

y.backward()
y_torch.backward()

assert np.allclose(y.data, y_torch.detach().numpy(), atol=1e-8)
assert np.allclose(x.grad.data, x_torch.grad.numpy(), atol=1e-8)

# ============================= 3D Tensor ==================================

x = np.random.randn(b, n, dim)
x_torch = torch.from_numpy(x).requires_grad_()
x = Parameter(x)

target = np.random.randint(0, n, size=(b, dim))
target_torch = torch.from_numpy(target)
target = Parameter(target)

y = D.nll_loss(x, target.data )
y_torch = F.nll_loss(x_torch, target_torch.long())

y.backward()
y_torch.backward()

assert np.allclose(y.data, y_torch.detach().numpy(), atol=1e-8)
assert np.allclose(x.grad.data, x_torch.grad.numpy(), atol=1e-8)
print("Success")
