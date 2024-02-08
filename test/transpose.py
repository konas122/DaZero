import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dazero import Variable
import dazero.functions as F


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
y.backward()
print(x.grad)

print(x.T)
print(x)
