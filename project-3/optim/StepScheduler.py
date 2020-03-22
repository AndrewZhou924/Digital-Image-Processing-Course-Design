from .Scheduler import Scheduler

class StepScheduler(Scheduler):
    def __init__(self, steps, base_lr, lr_decay, warmup_epoch=0, warmup_begin_lr=0.0, minimize_lr=0.0):
        super().__init__()

        self.lr = 0.

        self.steps = steps
        self.base_lr = base_lr
        self.lr_decay = lr_decay
        self.warmup_epoch = warmup_epoch
        self.warmup_begin_lr = warmup_begin_lr
        self.minimize_lr = minimize_lr

    def step(self, epoch):
        if epoch<self.warmup_epoch:
            self.lr = self.warmup_begin_lr + (self.base_lr - self.warmup_begin_lr) * (epoch+1) / self.warmup_epoch
        else:
            if epoch-self.warmup_epoch in self.steps:
                self.lr *= self.lr_decay
