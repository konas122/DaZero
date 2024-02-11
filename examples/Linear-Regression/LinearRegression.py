import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from header import *
from dazero import Parameter
import dazero.functions as F

lr = 0.1
iters = 100

def predict(x, W, b):
    y = F.matmul(x, W) + b
    return y


def mean_squared_loss(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)

    # plt.scatter(x, y)
    # plt.show()

    x, y = Parameter(x), Parameter(y)

    W = Parameter(np.random.randn(1, 1))
    b = Parameter(np.random.randn(1))

    for i in range(iters):
        y_hat = predict(x, W, b)
        loss = mean_squared_loss(y_hat, y)

        W.zero_grad()
        b.zero_grad()
        loss.backward()

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data
        print(W.data, b.data, loss.data, sep='\t')

    x_pred = np.linspace(0, 1, 100)
    y_pred = b.data + W.data * x_pred
    y_pred = y_pred.reshape((100,))

    plt.scatter(x.data, y.data)
    plt.plot(x_pred, y_pred)
    plt.show()
