import numpy as np

import dazero
from dazero import utils, cuda
from dazero.core import Function, Variable, as_variable, as_array


# ============================== NetWork ================================

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)

        axis = np.arange(x.ndim)
        i, j = x.ndim - 2, x.ndim - 1
        axis[i], axis[j] = axis[j], axis[i]
        x_T = x.transpose(*axis)

        gW = matmul(x_T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)


# ======================== Activation Function =========================

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        # y = 1 / (1 + xp.exp(-x))
        y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx

def relu(x):
    return ReLU()(x)


def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

def softmax(x, axis=1):
    return Softmax(axis)(x)


# ============================== Matrix ================================

class Dot(Function):
    """Matrix multiplication for 1D tensor"""
    def forward(self, x, W):
        f = lambda xs: (xs.ndim > 1 or xs.ndim == 2 and xs.shape[1] == 1)
        if f(x) or f(W):
            raise RuntimeError("1D tensors expected, but got wrong dimension tensors")
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = dot(gy, W.T)
        gW = dot(x.T, gy)
        return gx, gW

def dot(x, W):
    f = lambda xs: (xs.ndim > 1 or xs.ndim == 2 and xs.shape[1] != 1)
    if f(x) or f(W):
        raise RuntimeError("1D tensors expected, but got wrong dimension tensors")
    return Dot()(x, W)


class Matmul(Function):
    """Matrix multiplication for 2D and 3D tensor"""
    def forward(self, x, W):
        if x.ndim < 2 or W.ndim < 2:
            raise RuntimeError("Got wrong dimension tensors")
        xp = cuda.get_array_module(x)
        y = xp.matmul(x, W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)

        axis = np.arange(x.ndim)
        i, j = x.ndim - 2, x.ndim - 1
        axis[i], axis[j] = axis[j], axis[i]
        x_T = x.transpose(*axis)

        gW = matmul(x_T, gy)
        return gx, gW

def matmul(x, W):
    if x.ndim < 2 or W.ndim < 2:
        raise RuntimeError("Got wrong dimension tensors")
    return Matmul()(x, W)


class Bmm(Function):
    """Matrix multiplication for 3D tensor"""
    def forward(self, x, W):
        if x.ndim != 3 or W.ndim != 3:
            raise RuntimeError("3D tensors expected, but got wrong dimension tensors")
        xp = cuda.get_array_module(x)
        y = xp.matmul(x, W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = bmm(gy, W.transpose(0, 2, 1))
        gW = bmm(x.transpose(0, 2, 1), gy)
        return gx, gW

def bmm(x, W):
    if x.ndim != 3 or W.ndim != 3:
        raise RuntimeError("3D tensors expected, but got wrong dimension tensors")
    return Bmm()(x, W)


# ============================== Loss =================================

class MSELoss(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1

def mse_loss(x0, x1):
    return MSELoss()(x0, x1)


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)  # To avoid log(0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def sigmoid_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    x, t = as_variable(x), as_variable(t)
    N = len(x)
    p = sigmoid(x)
    p = clip(p, 1e-15, 1.0)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


def binary_cross_entropy(p, t):
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    N = len(t)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


# ========================== Tensor Operation =============================

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)

def transpose(x, axes=None):
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = sum(x, axis, keepdims)
    return y * (y.data.size / x.data.size)


mean = average


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to_utils(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)

def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


def expand_dims(x, axis):
    x = as_variable(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))


def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    return reshape(x, (x.shape[0], -1))


# ========================== accuracy / dropout ============================

def accuracy(y, t):
    """
    [WAR] This function is not differentiable.
    """
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))


def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if dazero.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    return x


# ================== batch_norm / embed_id / layer_norm =====================

class BatchNorm2d(Function):
    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.inv_std = None

    def forward(self, x, gamma, beta):
        assert x.ndim == 2 or x.ndim == 4

        x_ndim = x.ndim
        if x_ndim == 4:
            N, C, H, W = x.shape
            # (N, C, H, W) -> (N*H*W, C)
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        xp = cuda.get_array_module(x)

        if dazero.Config.train:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            xc = (x - mean) * inv_std

            m = x.size // gamma.size
            s = m - 1. if m - 1. > 1. else 1.
            adjust = m / s  # unbiased estimation
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var
            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std
        y = gamma * xc + beta

        if x_ndim == 4:
            # (N*H*W, C) -> (N, C, H, W)
            y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy):
        gy_ndim = gy.ndim
        if gy_ndim == 4:
            N, C, H, W = gy.shape
            gy = gy.transpose(0, 2, 3, 1).reshape(-1, C)

        x, gamma, beta = self.inputs
        batch_size = len(gy)

        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=0)
        ggamma = sum(xc * gy, axis=0)
        gx = gy - gbeta / batch_size - xc * ggamma / batch_size
        gx *= gamma * self.inv_std

        if gy_ndim == 4:
            gx = gx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return gx, ggamma, gbeta


def batch_norm(x, gamma=None, beta=None, mean=None, var=None, decay=0.9, eps=2e-5):
    """ `batch_norm` default use class `BatchNorm2d`. There is only this `BatchNorm2d`.
    """
    xp = cuda.get_array_module(x)
    D = x.shape[1]
    if mean is None:
        mean = xp.zeros(D, dtype=x.dtype)
    if var is None:
        var = xp.ones(D, dtype=x.dtype)
    if gamma is None:
        gamma = xp.ones(D, dtype=x.dtype)
    if beta is None:
        beta = xp.zeros(D, dtype=x.dtype)
    return BatchNorm2d(mean, var, decay, eps)(x, gamma, beta)


def embed_id(x, W):
    return W[x]


class LayerNorm(Function):
    def __init__(self, normalized_shape, mean, var, eps):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.axis = None
        self.axis_delta = None
        self.avg_mean = mean
        self.avg_var = var
        self.eps = eps
        self.inv_std = None

    def forward(self, x, gamma, beta):
        xp = cuda.get_array_module(x)
        x_shape = x.shape

        if self.axis is None:
            assert len(x_shape) > len(self.normalized_shape)
            self.axis = xp.arange(len(x_shape) - len(self.normalized_shape), len(x_shape))
            self.axis = tuple(list(self.axis))
            self.axis_delta = []
            for i in range(len(x_shape)):
                if i not in self.axis:
                    self.axis_delta.append(i)
            self.axis_delta = tuple(self.axis_delta)

        if dazero.Config.train:
            self.avg_mean = x.mean(axis=self.axis)
            self.avg_var = x.var(axis=self.axis)

            for _ in range(len(self.axis)):
                self.avg_mean = xp.expand_dims(self.avg_mean, -1)
                self.avg_var = xp.expand_dims(self.avg_var, -1)

            if len(gamma.shape) != len(x_shape):
                for _ in range(len(x_shape) - len(self.axis)):
                    gamma = xp.expand_dims(gamma, 0)
                    beta = xp.expand_dims(beta, 0)

            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std

            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std

        y = gamma * xc + beta
        return y

    def backward(self, gy):
        # previous layer delta
        xp = cuda.get_array_module(gy)
        normal_shape = xp.prod(self.normalized_shape)

        x, gamma, beta = self.inputs

        mean = self.avg_mean
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=self.axis_delta)
        ggamma = sum(xc * gy, axis=self.axis_delta)

        gx_1 = gamma * gy * self.inv_std
        gx_tmp = self.inv_std / normal_shape
        gx_2 = gx_tmp * sum(gy * gamma, axis=self.axis, keepdims=True)
        gx_3 = gx_tmp * xc * sum(gy * xc * gamma, axis=self.axis, keepdims=True)
        gx = gx_1 - gx_2 - gx_3

        return gx, ggamma, gbeta


def layer_norm(x, normalized_shape, gamma=None, beta=None, mean=None, var=None, eps=1e-5):
    xp = cuda.get_array_module(x)
    D = x.shape[1]
    S = x.shape[1:]
    if mean is None:
        mean = xp.zeros(D, dtype=x.dtype)
    if var is None:
        var = xp.ones(D, dtype=x.dtype)
    if gamma is None:
        gamma = xp.ones(S, dtype=x.dtype)
    if beta is None:
        beta = xp.zeros(S, dtype=x.dtype)
    return LayerNorm(normalized_shape, mean, var, eps)(x, gamma, beta)


# ============================= Basic functions =============================

class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx

def log(x):
    return Log()(x)


# ===========================================================================

class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()  # weakref

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)

def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx

def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


# ==========================================================================

from dazero.functions_conv import conv2d
from dazero.functions_conv import deconv2d
from dazero.functions_conv import conv2d_simple
from dazero.functions_conv import im2col
from dazero.functions_conv import col2im
from dazero.functions_conv import pooling_simple
from dazero.functions_conv import max_pooling
from dazero.functions_conv import average_pooling

from dazero.core import add
from dazero.core import sub
from dazero.core import rsub
from dazero.core import mul
from dazero.core import div
from dazero.core import neg
from dazero.core import pow
