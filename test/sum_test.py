import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dazero import Variable
import dazero.functions as F


np.random.seed(0)
x = Variable(np.array([1, 2, 3, 4, 5, 6]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2, 3, 4, 5))
y = F.sum(x, keepdims=True)
print(y.shape)
