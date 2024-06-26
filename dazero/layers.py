import os
import weakref
import numpy as np

from dazero import cuda
import dazero.functions as F
from dazero.utils import pair
from dazero.core import Tensor


class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Tensor, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def zero_grad(self):
        for param in self.params():
            param.zero_grad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        if cuda.gpu_enable == False:
            return
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, param_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(param_dict, key)
            else:
                param_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise e

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


# ============================== Pooling ====================================

class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        return F.max_pooling(x, self.kernel_size, self.stride, self.pad)


class AvgPool2d(Layer):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        return F.average_pooling(x, self.kernel_size, self.stride, self.pad)


# =============================================================================
# Linear / Conv2d / Deconv2d
# =============================================================================

class Linear(Layer):
    def __init__(self, in_size=None, out_size=None, bias=True, dtype=np.float32):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Tensor(None, name='W')
        self._init_W()

        if bias:
            self.b = Tensor(np.zeros(out_size, dtype=dtype), name='b')
        else:
            self.b = None

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, pad=0, bias=True, dtype=np.float32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Tensor(None, name='W')
        self._init_W()

        if bias:
            self.b = Tensor(np.zeros(out_channels, dtype=dtype), name='b')
        else:
            self.b = None

    def _init_W(self):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = np.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Deconv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, pad=0, bias=True, dtype=np.float32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Tensor(None, name='W')
        self._init_W()

        if bias:
            self.b = Tensor(np.zeros(out_channels, dtype=dtype), name='b')
        else:
            self.b = None

    def _init_W(self):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = np.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        y = F.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y


# =============================================================================
# Embedding / BatchNorm2d / LayerNorm
# =============================================================================

class Embedding(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = Tensor(np.random.randn(in_size, out_size), name='W')

    def __call__(self, x):
        y = self.W[x]
        return y


class BatchNorm2d(Layer):
    def __init__(self):
        super().__init__()
        # `.avg_mean` and `.avg_var` are `Tensor` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Tensor(None, name='avg_mean')
        self.avg_var = Tensor(None, name='avg_var')
        self.gamma = Tensor(None, name='gamma')
        self.beta = Tensor(None, name='beta')

    def _init_params(self, x):
        xp = cuda.get_array_module(x)
        D = x.shape[1]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def __call__(self, x):
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.batch_norm(x, self.gamma, self.beta, self.avg_mean.data, self.avg_var.data)


class LayerNorm(Layer):
    def __init__(self, normalized_shape, gamma=None, beta=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        # `.avg_mean` and `.avg_var` are `Tensor` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Tensor(None, name='avg_mean')
        self.avg_var = Tensor(None, name='avg_var')
        self.gamma = Tensor(gamma, name='gamma')
        self.beta = Tensor(beta, name='beta')

    def _init_params(self, x):
        xp = cuda.get_array_module(x)
        D = x.shape[1]
        S = x.shape[1:]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(S, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(S, dtype=x.dtype)

    def __call__(self, x):
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.avg_mean.data, self.avg_var.data)


# =============================================================================
# RNN / LSTM
# =============================================================================

class RNN(Layer):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.x2h = Linear(in_size,     hidden_size)
        self.h2h = Linear(hidden_size, hidden_size, bias=False)
        self.h = None   # h_{t-1}

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new


class LSTM(Layer):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        I, H = in_size, hidden_size

        self.x2f = Linear(I, H)
        self.x2i = Linear(I, H)
        self.x2o = Linear(I, H)
        self.x2u = Linear(I, H)

        self.h2f = Linear(H, H, bias=False)
        self.h2i = Linear(H, H, bias=False)
        self.h2o = Linear(H, H, bias=False)
        self.h2u = Linear(H, H, bias=False)

        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(   self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = F.tanh(   self.x2u(x) + self.h2u(self.h))

        if self.c is None:
            c_new = i * u
        else:
            c_new = (f * self.c) + (i * u)
        h_new = o * F.tanh(c_new)

        self.h, self.c = h_new, c_new
        return h_new
