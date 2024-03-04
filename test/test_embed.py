import unittest
import numpy as np
from dazero import Variable
from dazero.functions import embedding


class TestEmbedID(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array([[1, 2, 3], [2, 3, 4]]))
        W = Variable(np.eye(5))
        y = embedding(x, W)

        expected = np.array(
            [[[0, 1, 0, 0, 0 ],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0]],
             [[0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]]]
        )
        self.assertTrue(np.alltrue(y.data == expected))

    def test_backward(self):
        x = Variable(np.array([[1, 2, 3], [2, 3, 4]]))
        W = Variable(np.eye(5))
        y = embedding(x, W)
        y.backward()

        expected = np.array(
            [[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
             [1, 1, 1, 1, 1]], dtype=np.float64)

        self.assertTrue(np.alltrue(W.grad.data == expected))
