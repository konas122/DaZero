import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dazero import Variable
from dazero import Model
import dazero.layers as L
import dazero.functions as F


lr = 0.2
iters = 10000

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


class Network(Model):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(in_size, hidden_size)
        self.l2 = L.Linear(hidden_size, out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        return self.l2(y)


model = Network(1, 100, 1)
for i in range(iters):
    y_pred = model(x)
    loss = F.mse_loss(y, y_pred)

    model.zero_grad()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss.data)


model.plot(x)
