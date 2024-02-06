from header import *
from dazero.core import Variable, Add


if __name__ == "__main__":
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    f = Add()
    y = f(x0, x1)
    print(y.data)
