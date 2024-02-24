import unittest
import numpy as np
import dazero
import dazero.functions as F
from dazero import Model
from dazero.utils import array_allclose


class Layer(Model):
    def __init__(self):
        super().__init__()
        self.layer = dazero.layers.Linear(in_size=2, out_size=3, nobias=True)
    
    def forward(self, inputs):
        return self.layer(inputs)


class TestWeightDecay(unittest.TestCase):
    def test_compare1(self):
        rate = 0.4
        x = np.random.rand(10, 2)
        t = np.zeros((10)).astype(int)
        model = Layer()
        model.layer.W.data = np.ones_like(model.layer.W.data)
        optimizer = dazero.optimizers.SGD(model)
        optimizer.add_hook(dazero.optimizers.WeightDecay(rate=rate))

        model.zero_grad()
        y = model(x)
        y = F.softmax_cross_entropy(y, t)
        y.backward()
        optimizer.step()
        W0 = model.layer.W.data.copy()

        model.layer.W.data = np.ones_like(model.layer.W.data)
        optimizer.hooks.clear()
        model.zero_grad()
        y = model(x)
        y = F.softmax_cross_entropy(y, t) + rate / 2 * (model.layer.W ** 2).sum()
        y.backward()
        optimizer.step()
        W1 = model.layer.W.data
        self.assertTrue(array_allclose(W0, W1))
