import os
try:
    import torch
except ImportError:
    os._exit(0)
import numpy as np
import torch.nn.functional as F

from dazero import Variable
import dazero.functions as D


dim = 1
np.random.seed(0)
x = np.random.randn(40, 3, 2)
x_torch = torch.from_numpy(x).requires_grad_()
x = Variable(x)

y = D.log_softmax(x, axis=dim)
y_torch = F.log_softmax(x_torch, dim=dim)

grad = np.random.randn(*y.shape)
y.grad = Variable(grad)
y.backward()
y_torch.backward(torch.from_numpy(grad))

assert np.allclose(y.data, y_torch.detach().numpy(), atol=1e-8)
assert np.allclose(x.grad.shape, x_torch.grad.shape, atol=1e-6)
print("Success")
