import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dazero import Variable
import dazero.functions as F

x = Variable(np.random.randn(2))
W = Variable(np.random.randn(2))
y = F.dot(x, W)

y.backward()
print(x.grad.shape == x.shape)
print(W.grad.shape == W.shape)
print(W.data == x.grad.data)
print(x.data == W.grad.data)
