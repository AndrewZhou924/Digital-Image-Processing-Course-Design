import numpy as np
from .Layer import Layer

class BatchNorm(Layer):
    def __repr__(self):
        return 'BatchNorm'

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.train = True
        self.eps = eps
        self.momentum = momentum

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.parameters = [
            np.random.randn(num_features),
            np.random.randn(num_features)
        ]
        self.dx = None
        self.dgamma = None
        self.dbeta = None

        self.cache = None
        self.batch_size = None

    def forward(self, x):
        self.batch_size = x.shape[0]

        if self.train:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            self.running_mean = self.running_mean * self.momentum + mean * (1 - self.momentum) # Moving average for mean
            self.running_var = self.running_var * self.momentum + var * (1 - self.momentum) # Moving average for var

            x_ = (x - mean) / np.sqrt(var + self.eps) # Normalization
            y = x_ * self.parameters[0] + self.parameters[1] # Scale and Shift

            self.cache = (x_, var)
        else:
            y = (x - self.running_mean) / np.sqrt(self.running_var + self.eps) * self.parameters[0] + self.parameters[1]
        return y

    def backward(self, dy, *args, **kwargs):
        if self.train:
            x_, var = self.cache

            self.dgamma = np.sum(dy * x_, axis=0)
            self.dbeta = np.sum(dy, axis=0)

            dx_ = dy * self.parameters[0]
            dx = self.batch_size * dx_ - np.sum(dx_, axis=0) - x_ * np.sum(dx_ * x_, axis=0)
            self.dx = dx * (1.0 / self.batch_size) / np.sqrt(self.eps + var)
        else:
            self.dx = dy

        return self.dx

    def update(self, optimizer):
        self.parameters[0] = optimizer.update(self.parameters[0], self.dgamma, self.batch_size, id(self), 0)
        self.parameters[1] = optimizer.update(self.parameters[1], self.dbeta, self.batch_size, id(self), 1)