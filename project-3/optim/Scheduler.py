class Scheduler:
    def __init__(self):
        self.lr = 0.

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, optimizer):
        optimizer.lr = self.lr
