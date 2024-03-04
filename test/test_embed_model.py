import unittest
import numpy as np
from dazero import Variable
from dazero.models import Sequential
from dazero.layers import Embedding, Linear
from dazero.functions import sigmoid, tanh, mse_loss
from dazero.optimizers import SGD


class TestEmbedModel(unittest.TestCase):

    def test_train(self):
        x = Variable(np.array([1, 2, 1, 2]))
        target = Variable(np.array([[0], [1], [0], [1]]))

        model = Sequential(
            Embedding(5, 3),
            tanh,
            Linear(3, 1),
            sigmoid
        )

        optimizer = SGD(model, lr=0.5)

        np.random.seed(0)
        model[0].W.data = np.random.rand(5, 3)
        model[2].W.data = np.random.rand(3, 1)

        log = []
        for _ in range(10):
            pred = model(x)
            loss = mse_loss(pred, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            log.append(loss.data)

        expected = np.array([
            0.25458621375800417,
            0.24710456626288174,
            0.24017425722643587,
            0.23364699169761943,
            0.22736806682064464,
            0.2211879225084124,
            0.2149697611450082,
            0.20859448689275056,
            0.2019642998089552,
            0.195005940360243
            ])

        res = np.allclose(np.array(log), expected)
        self.assertTrue(res)
