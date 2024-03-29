import unittest
import numpy as np
from dazero import Variable
import dazero.functions as F
from dazero.utils import gradient_check


class TestTranspose(unittest.TestCase):
    def test_forward1(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.transpose(x)
        self.assertEqual(y.shape, (3, 2))

    def test_backward1(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(gradient_check(F.transpose, x))

    def test_backward2(self):
        x = np.array([1, 2, 3])
        self.assertTrue(gradient_check(F.transpose, x))

    def test_backward3(self):
        x = np.random.randn(10, 5)
        self.assertTrue(gradient_check(F.transpose, x))

    def test_backward4(self):
        x = np.array([1, 2])
        self.assertTrue(gradient_check(F.transpose, x))
