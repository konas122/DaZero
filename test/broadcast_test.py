import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dazero import Variable
import dazero.functions as F


x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))

y = x1 + x0
assert np.array_equal(y.shape, x0.shape)

y.backward()
assert np.array_equal(x0.shape, x0.grad.shape)
assert np.array_equal(x1.shape, x1.grad.shape)
print("Success")
