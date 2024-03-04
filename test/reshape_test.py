import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dazero import Variable
import dazero.functions as F


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

np.random.seed(0)
x = Variable(np.random.randn(1, 2, 3))
y = x.reshape((2, 3))
print(y.shape)
y = x.reshape([2, 3])
print(y.shape)
y = x.reshape(2, 3)
print(y.shape)
