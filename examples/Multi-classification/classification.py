import math
import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from header import *

import dazero
from dazero import Model
from dazero import optimizers
import dazero.layers as L
import dazero.functions as F


# Hyperparameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0


class MLP(Model):
    def __init__(self, in_dims, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []
        self.in_dims = in_dims

        for i, out_size in enumerate(fc_output_sizes):
            if i == 0:
                layer = L.Linear(in_dims, out_size)
            else:
                layer = L.Linear(fc_output_sizes[i - 1], out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


x, t = dazero.datasets.get_spiral(train=True)

model = MLP(x.shape[1], (hidden_size, 3))
optimizer = optimizers.SGD(model, lr)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # Shuffle index for data
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += float(loss.data) * len(batch_t)

    # Print loss every epoch
    if epoch % 10 == 0:
        avg_loss = sum_loss / data_size
        print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))

# Plot boundary area the model predict
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

with dazero.no_grad():
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

# Plot data points of the dataset
N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0], x[i][1], s=40,  marker=markers[c], c=colors[c])
plt.show()
