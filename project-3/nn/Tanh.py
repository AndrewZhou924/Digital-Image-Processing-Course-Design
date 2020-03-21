import numpy as np
from .Layer import Layer

class Tanh(Layer):
    def __repr__(self):
        return 'Tanh'

    def __init__(self):
        super().__init__()
        self.dx = None

    def forward(self, x):
        y = np.tanh(x)
        self.dx = 1 - y**2 # local_grad
        return y

    def backward(self, dy, *args, **kwargs):
        self.dx = dy * self.dx
        return self.dx