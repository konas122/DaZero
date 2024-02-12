import numpy as np

from dazero import Layer
import dazero.functions as F
import dazero.layers as L
from dazero import utils


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)
