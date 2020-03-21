import numpy as np
from .Layer import Layer

class CrossEntropyLoss(Layer):
    def __repr__(self):
        return 'CrossEntropyLoss'

    def __init__(self):
        super().__init__()
        self.dx = None

    def forward(self, inputs, targets):
        inputs_ = inputs - np.max(inputs)
        pred = self.softmax(inputs_)
        loss = np.sum(-np.log(pred[range(inputs_.shape[0]), targets])) / inputs_.shape[0]
        self.dx = pred
        self.dx[range(inputs_.shape[0]), targets] -= 1
        return loss

    def backward(self, *args, **kwargs):
        return self.dx

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

if __name__=='__main__':
    x = np.array([[-1, -1, 1]])
    y = np.array([2], dtype=np.int32)
    criterion = CrossEntropyLoss()
    loss = criterion(x, y)
    print(loss)
    print(criterion.grad)