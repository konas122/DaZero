import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dazero import utils
from dazero import Variable
import dazero.functions as F


x = Variable(np.array(1.0))
y = F.tanh(x)

x.name = 'x'
y.name = 'y'
y.backward(retain_graph=True)

iters = 0
for i in range(iters):
    gx = x.grad
    x.zero_grad()
    gx.backward(retain_graph=True)

gx = x.grad
gx.name = 'gx' + str(iters+1)
utils.plot_dot_graph(gx, verbose=False, to_file='tanh.png')
