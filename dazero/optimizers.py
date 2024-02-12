import math
import numpy as np

from dazero import Model, Parameter


class Optimizer:

    def __init__(self, model):
        self.target = None
        self.hooks = []
        self._setup(model)

    def _setup(self, target):
        if not isinstance(target, Model):
            raise TypeError("It must be of type dazero.Model")
        self.target = target
        return self
    
    def step(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)
        
        for param in params:
            self.step_one(param)
    
    def step_one(self):
        raise NotImplementedError()
    
    def add_hook(self, f):
        self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, model, lr=0.01):
        super().__init__(model)
        self.lr = lr
    
    def step_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.9):
        super().__init__(model)
        self.lr = lr
        self.momentum = momentum
        self.vs = {}
    
    def step_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)
        
        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):
    def __init__(self, model, lr=0.01, eps=1e-8):
        super().__init__(model)
        self.lr = lr
        self.eps = eps
        self.hs = {}
    
    def step_one(self, param):
        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = np.zeros_like(param.data)
        
        lr = self.lr
        eps = self.eps
        grad = param.grad.data
        h = self.hs[h_key]

        h += grad * grad
        param.data -= lr * grad / (np.sqrt(h) + eps)
    

class AdaDelta(Optimizer):
    def __init__(self, model, rho=0.95, eps=1e-6):
        super().__init__(model)
        self.rho = rho
        self.eps = eps
        self.msg = {}
        self.msdx = {}

    def step_one(self, param):
        key = id(param)
        if key not in self.msg:
            self.msg[key] = np.zeros_like(param.data)
            self.msdx[key] = np.zeros_like(param.data)

        msg, msdx = self.msg[key], self.msdx[key]
        rho = self.rho
        eps = self.eps
        grad = param.grad.data

        msg *= rho
        msg += (1 - rho) * grad * grad
        dx = np.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx *= rho
        msdx += (1 - rho) * dx * dx
        param.data -= dx


class Adam(Optimizer):
    def __init__(self, model, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(model)
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def step(self, *args, **kwargs):
        self.t += 1
        super().step(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def step_one(self, param):
        key = id(param)
        if key not in self.ms:
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (np.sqrt(v) + eps)