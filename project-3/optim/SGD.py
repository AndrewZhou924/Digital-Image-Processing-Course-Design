import numpy as np
from .Optim import Optim
class SGD(Optim):
    def __init__(self, parameters, lr=1e-2, momentum=0):
        super().__init__()

        self.lr = lr
        self.momentum = momentum
        self.m = {}

        for k, v in parameters.items():
            if self.m.get(k, 0)==0:
                self.m[k] = []
            for param in v:
                self.m[k].append(np.zeros(param.shape))

    def update(self, w, dw, batch_size, layerid, index):
        self.m[layerid][index] = self.momentum * self.m[layerid][index] + dw / batch_size
        return w - self.lr * self.m[layerid][index]