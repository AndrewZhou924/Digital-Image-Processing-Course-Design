import numpy as np
from .Optim import Optim
class SGD(Optim):
    def __init__(self, parameters, lr=1e-2, momentum=0, weight_decay=0):
        super().__init__()

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.m = {}

        for k, v in parameters.items():
            if self.m.get(k, 0)==0:
                self.m[k] = []
            for param in v:
                self.m[k].append(np.zeros(param.shape))

    def update(self, w, dw, batch_size, layerid, index):
        dw += self.weight_decay * w
        dw /= batch_size
        self.m[layerid][index] = self.momentum * self.m[layerid][index] + self.lr * dw
        return w - self.m[layerid][index]