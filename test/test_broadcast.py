import unittest
import numpy as np
from dazero import Variable
import dazero.functions as F


class TestBroadcast(unittest.TestCase):
    def test_shape_check(self):
        x = Variable(np.random.randn(1, 10))
        b = Variable(np.random.randn(10))
        y = x + b
        loss = F.sum(y)
        loss.backward()
        self.assertEqual(b.grad.shape, b.shape)
