import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dazero import Variable
import dazero.functions as F


x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))

y = x1 + x0
print(y)

y.backward()
print(x0.grad)
print(x1.grad)
