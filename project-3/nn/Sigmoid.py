import numpy as np
from .Layer import Layer
class Sigmoid(Layer):
    def __repr__(self):
        return 'Sigmoid'

    def __init__(self):
        super().__init__()
        self.dx = None

    def forward(self, x):
        y = 1.0 / (1.0 + np.exp(-x))
        self.dx = y * (1 - y)
        return y

    def backward(self, dy, *args, **kwargs):
        self.dx = self.dx * dy
        return self.dx