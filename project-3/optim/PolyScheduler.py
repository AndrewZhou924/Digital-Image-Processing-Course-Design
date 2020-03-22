from .Scheduler import Scheduler

class PolyScheduler(Scheduler):
    def __init__(self, epoch_num, base_lr, power=2, warmup_epoch=0, final_lr=0.0, warmup_begin_lr=0.0):
        super().__init__()

        self.lr = 0.

        self.epoch_num = epoch_num
        self.base_lr = base_lr
        self.power = power
        self.warmup_epoch = warmup_epoch
        self.final_lr = final_lr
        self.warmup_begin_lr = warmup_begin_lr

    def step(self, epoch):
        if epoch<self.warmup_epoch:
            self.lr = self.warmup_begin_lr + (self.base_lr - self.warmup_begin_lr) * (epoch+1) / self.warmup_epoch
        else:
            self.lr = self.final_lr + (self.base_lr - self.final_lr) * pow(1-float(epoch+1-self.warmup_epoch)/float(self.epoch_num), self.power) if epoch<self.epoch_num else 0