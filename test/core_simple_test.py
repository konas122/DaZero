if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import Variable

import numpy as np

x = Variable(np.array(3.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)
