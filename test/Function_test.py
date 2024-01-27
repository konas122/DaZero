from header import *
from src.core_simple import Variable, Function


class Square(Function):
    def forward(self, x):
        return x ** 2


if __name__ == "__main__":
    x = Variable(np.array(10))
    f = Square()
    y = f(x)

    print(type(x))
    print(y.data)

    f1 = Square()
    print(f1(y).data)
