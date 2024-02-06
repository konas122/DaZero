from header import *
from dazero import Variable


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    y = f(x)
    x.zero_grad()
    y.backward(create_graph=True)

    gx = x.grad
    x.zero_grad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
