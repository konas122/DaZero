import unittest
import numpy as np
from dazero import Variable
import dazero.functions as F


class TestSoftmaxCrossEntropy(unittest.TestCase):

    def test_forward(self):
        x = Variable(np.array([[-1, 0, 1, 2], [2, 0, 1, -1]]))
        t = Variable(np.array([3,0]))
        y = F.softmax_cross_entropy(x, t)

        expected = 0.440189
        self.assertTrue(abs(y.data - expected) < 1e-6)

    def test_backward(self):
        x = Variable(np.array([[-1, 0, 1, 2], [2, 0, 1, -1]]))
        t = Variable(np.array([3, 0]))
        y = F.softmax_cross_entropy(x, t)
        y.backward()

        #print(x.grad)
        expected = np.array([[ 0.0160293 ,  0.04357216,  0.11844141, -0.17804287],
       [-0.17804287,  0.04357216,  0.11844141,  0.0160293 ]])
        #print(x.grad - expected)
        self.assertTrue(np.allclose(x.grad.data, expected, atol=1e-8))
