import numpy as np
from .Layer import Layer

class ReLU(Layer):
    def __repr__(self):
        return 'ReLU'

    def __init__(self):
        super().__init__()

        self.dx = None
        self.mask = None

    def forward(self, x):
        self.mask = np.array(x > 0, dtype=np.int32)
        return self.mask * x

    def backward(self, dy, *args, **kwargs):
        self.dx = self.mask * dy
        return self.dx
