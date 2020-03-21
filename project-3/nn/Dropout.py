import numpy as np
from .Layer import Layer

class Dropout(Layer):
    def __repr__(self):
        return 'Dropout'

    def __init__(self, p):
        super().__init__()

        self.p = p
        self.dx = None
        self.train = True

    def forward(self, x):
        if self.train:
            self.mask = np.random.binomial(1, 1-self.p, x.shape) / (1-self.p)
            return x * self.mask
        else:
            return x

    def backward(self, dy, *args, **kwargs):
        self.dx = dy * self.mask
        return self.dx

if __name__=='__main__':
    x = np.random.randn(2, 10)
    print(x)
    print(x/0.7)
    dropout = Dropout(0.3)
    y = dropout(x)
    print(y)
    dy = np.random.randn(2, 10)
    dx = dropout.backward(dy)
    print(dy)
    print(dy/0.7)
    print(dx)