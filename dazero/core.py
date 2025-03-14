import weakref
import contextlib
import numpy as np
from queue import PriorityQueue

import dazero


class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config("enable_backprop", False)

def test_mode():
    return using_config("train", False)


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# ============================== Variable ===================================

try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None) -> None:
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{} is not supported!'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, retain_graph=False):
        if self.grad is None:
            xp = dazero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = PriorityQueue()
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                seen_set.add(f)
                funcs.put(dazero.utils.BackwardFun(f))

        add_func(self.creator)

        while not funcs.empty():
            f = funcs.get().fun
            gys = [output().grad for output in f.outputs]

            with using_config('enable_backprop', retain_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def detach(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x._detach()

    def _detach(self):
        self.creator = None

    def zero_grad(self):
        self.grad = None

    def sum(self, axis=None, keepdims=False):
        return dazero.functions.sum(self, axis, keepdims)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dazero.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dazero.functions.transpose(self, axes)

    @property
    def T(self):
        return dazero.functions.transpose(self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def to_cpu(self):
        if self.data is not None:
            self.data = dazero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if dazero.cuda.gpu_enable == False:
            return
        if self.data is not None:
            self.data = dazero.cuda.as_cupy(self.data)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = dazero.functions.get_item

    Variable.matmul = dazero.functions.matmul
    Variable.dot = dazero.functions.dot
    Variable.max = dazero.functions.max
    Variable.min = dazero.functions.min
    Variable.mean = dazero.functions.mean


class Tensor(Variable):
    pass


# =============================== Function ==================================

class Function:

    def __init__(self):
        pass

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


# ============================ Basic Operation ==============================

class Add(Function):
    def __init__(self):
        super().__init__()

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = dazero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dazero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

def add(x0, x1):
    x1 = as_array(x1, dazero.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
    def __init__(self):
        super().__init__()

    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dazero.functions.sum_to(gx0, x0.shape)
            gx1 = dazero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

def mul(x0, x1):
    x1 = as_array(x1, dazero.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)


class Sub(Function):
    def  __init__(self):
        super().__init__()

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = dazero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dazero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

def sub(x0, x1):
    x1 = as_array(x1, dazero.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1, dazero.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


class Div(Function):
    def __init__(self):
        super().__init__()

    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dazero.functions.sum_to(gx0, x0.shape)
            gx1 = dazero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1, dazero.cuda.get_array_module(x0.data))
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1, dazero.cuda.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)
