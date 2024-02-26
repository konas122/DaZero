import os
try:
    import torch
except:
    os._exit(0)
import numpy as np
import torch.nn as nn

import dazero.layers as L
from dazero import Variable


in_size, out_size = 23, 6
pos_emb = L.Embedding(in_size, out_size)
pos_emb_torch = nn.Embedding(in_size, out_size)
pos_emb.W.data = pos_emb_torch.weight.data.numpy()

t = 22
inputs = np.arange(t)
inputs_torch = torch.arange(t)
inputs = Variable(inputs)

position = pos_emb(inputs)
position_torch = pos_emb_torch(inputs_torch)

grad = np.random.randn(*position.shape)
position.grad = Variable(grad)
position.backward()
position_torch.backward(torch.from_numpy(grad))

assert position.shape == position_torch.shape
assert np.allclose(position.data, position_torch.detach().numpy(), atol=1e-6)
assert np.allclose(pos_emb.W.grad.data, pos_emb_torch.weight.grad.numpy(), atol=1e-6)

print("Success")
