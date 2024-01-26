import numpy as np


class Variable:

    def __init__(self, data) -> None:
        self.data = data


class Function:
    
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()

