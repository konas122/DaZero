import os
import numpy as np

try:
    import torch
except ImportError:
    os._exit(0)

from dazero import Tensor
import dazero.functions as F


input = np.random.randn(2, 3, 4)
mat2 = np.random.randn(2, 4, 5)

# print(np.dot(input, mat2).shape)
# print(np.matmul(input, mat2).shape)

input_torch, mat2_torch = torch.from_numpy(input).requires_grad_(), torch.from_numpy(mat2).requires_grad_()
res_torch = torch.bmm(input_torch, mat2_torch)

input = Tensor(input)
mat2 = Tensor(mat2)
res = F.bmm(input, mat2)
assert np.allclose(res.data, res_torch.detach().numpy(), atol=1e-9) == True

res.backward(retain_grad=True)
res_torch.backward(torch.ones_like(res_torch))
# print(input_torch.grad.numpy())

assert np.allclose(input_torch.grad.numpy(), input.grad.data, atol=1e-9) == True
assert np.allclose(mat2_torch.grad.numpy(), mat2.grad.data, atol=1e-9) == True
print("Success")
