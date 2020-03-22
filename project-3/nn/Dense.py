import numpy as np
from .Layer import Layer

class Dense(Layer):
    def __repr__(self):
        return 'Dense'

    def __init__(self, in_features, out_features, cuda=False):
        super().__init__()

        self.parameters = [
            np.random.randn(in_features, out_features),
            np.random.randn(out_features)
        ]

        self.dx = None
        self.dw = None
        self.db = None

        self.batch_size = None

    def forward(self, x):
        self.batch_size = x.shape[0]
        y = np.dot(x, self.parameters[0]) + self.parameters[1]

        self.dw = np.transpose(x)
        self.db = np.zeros(self.parameters[1].shape)
        return y

    def backward(self, dy, *args, **kwargs):
        self.dx = np.dot(dy, self.parameters[0].transpose())

        self.dw = np.dot(self.dw, dy)

        self.db = np.sum(dy, axis=0)
        return self.dx

    def update(self, optimizer):
        self.parameters[0] = optimizer.update(self.parameters[0], self.dw, self.batch_size, id(self), 0)
        self.parameters[1] = optimizer.update(self.parameters[1], self.db, self.batch_size, id(self), 1)
