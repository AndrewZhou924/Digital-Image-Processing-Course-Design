'''VGG11/13/16/19 in Jittor.'''
import jittor as jt
import jittor.nn as nn
import numpy as np


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def execute(self, x):
        out = self.features(x)
        out = jt.reshape(out, [out.shape[0], -1])
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.Pool(kernel_size=2, stride=2, op="maximum")]
            else:
                layers += [nn.Conv(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm(x),
                           nn.ReLU()]
                in_channels = x
        layers += [nn.Pool(kernel_size=1, stride=1, op="mean")]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    print(net)
    x = jt.array(np.random.randn(3, 3, 32, 32), dtype=np.float32)
    y = net(x)
    print(y.shape)

if __name__=='__main__':
    test()