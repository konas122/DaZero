from header import *
from src.core_simple import Variable, Function


class Square(Function):

    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):

    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)


if __name__ == "__main__":
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
