from .Optim import Optim
import numpy as np

class Adam(Optim):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4):
        super().__init__()

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.steps = 0

        self.m = {}
        self.v = {}

        for k, v in parameters.items():
            if self.m.get(k, 0) == 0:
                self.m[k] = []
                self.v[k] = []
            for param in v:
                self.m[k].append(np.zeros(param.shape))
                self.v[k].append(np.zeros(param.shape))

    def update(self, w, dw, batch_size, layerid, index):
        self.steps += 1
        dw += w * self.weight_decay
        dw = dw / batch_size

        self.m[layerid][index] = self.betas[0] * self.m[layerid][index] + (1 - self.betas[0]) * dw
        self.v[layerid][index] = self.betas[1] * self.v[layerid][index] + (1 - self.betas[1]) * dw * dw

        m_ = self.m[layerid][index] / (1 - self.betas[0] ** self.steps)
        v_ = self.v[layerid][index] / (1 - self.betas[1] ** self.steps)

        return w - self.lr * m_ / (v_ ** 0.5 + self.eps)