from header import *
from dazero.core import Variable, Add


if __name__ == "__main__":
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    f = Add()
    y = f(x0, x1)
    print(y.data)
    print('----------')

    x0 = Variable(np.random.randn(2, 3))
    x1 = Variable(np.random.randn(1))
    y = f(x0, x1)
    print(x0.data)
    print(x1.data, end='\n\n')
    print(y.data, end='\n\n')
    y.backward()
    print(x0.grad)
    print(x1.grad)
