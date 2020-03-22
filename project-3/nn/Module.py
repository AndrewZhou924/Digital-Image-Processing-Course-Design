class Module:
    def __init__(self):
        self.layers = []
        self.parameters_ = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for i in range(len(self.layers)-1, -1, -1):
            if i==len(self.layers)-1:
                self.layers[i].backward(grad)
            else:
                self.layers[i].backward(self.layers[i+1].dx)

    def update(self, optimizer):
        for layer in self.layers:
            layer.update(optimizer)

    def parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                self.parameters_[id(layer)] = layer.parameters
        return self.parameters_

    def train(self, *args, **kwargs):
        for layer in self.layers:
            if layer.__repr__() in ['BatchNorm', 'Dropout']:
                layer.train = True

    def eval(self, *args, **kwargs):
        for layer in self.layers:
            if layer.__repr__() in ['BatchNorm', 'Dropout']:
                layer.train = False