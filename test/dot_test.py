import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dazero import Variable
import dazero.functions as F

x = Variable(np.random.randn(2))
W = Variable(np.random.randn(2))
y = F.dot(x, W)

y.backward()
assert x.grad.shape == x.shape
assert W.grad.shape == W.shape
assert np.allclose(W.data, x.grad.data, atol=1e-8) == True
assert np.allclose(x.data, W.grad.data, atol=1e-8) == True
print("Success")
