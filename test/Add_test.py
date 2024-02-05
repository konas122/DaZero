from header import *
from dazero.core_simple import Variable, Function


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


if __name__ == "__main__":
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    f = Add()
    y = f(x0, x1)
    print(y.data)
