'''
The xresnet archtecture as presented on the bag of tricks paper
'''
import torch
import torch.nn as nn
from functools import partial


def noop(x): return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

act_fn = nn.ReLU(inplace=True)

def init_cnn(m):
    if getattr(m, 'bias', None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for l in m.children():
        init_cnn(l)

def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    if act:
        layers.append(act_fn)
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    '''
    The residual block
    '''
    def __init__(self, expansion, ni, nh, stride=1):
        super().__init__()
        nf, ni = nh*expansion, ni*expansion
        layers  = [conv_layer(ni, nh, 3, stride=stride),
                   conv_layer(nh, nf, 3, zero_bn=True, act=False)
        ] if expansion == 1 else [
                   conv_layer(ni, nh, 1),
                   conv_layer(nh, nh, 3, stride=stride),
                   conv_layer(nh, nf, 1, zero_bn=True, act=False)
        ]
        self.convs = nn.Sequential(*layers)
        self.idconv = noop if ni == nf else conv_layer(ni, nf, 1, act=False)
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return act_fn(self.convs(x) + self.idconv(self.pool(x)))

class XResNet(nn.Sequential):
    '''
    The XResnet
    '''
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000):
        nfs = [c_in, (c_in+1)*8, 64, 64]
        stem = [conv_layer(nfs[i], nfs[i+1], stride=2 if i == 0 else 1)
                for i in range(3)]

        nfs = [64//expansion, 64, 128, 256, 512]
        res_layers = [cls._make_layer(expansion, nfs[i], nfs[i+1],
                                      n_blocks=l, stride=1 if i == 0 else 2)
                      for i, l in enumerate(layers)]
        res = cls(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *res_layers,
            nn.AdaptiveAvgPool2d(1), Flatten(),
            nn.Linear(nfs[-1]*expansion, c_out),
        )
        init_cnn(res)
        return res

    @staticmethod
    def _make_layer(expansion, ni, nf, n_blocks, stride):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i == 0 else nf, nf, stride if i == 0 else 1)
              for i in range(n_blocks)])

    def _set_grad(self, m, b):
        '''
        Internal method for freezing and unfreezing encoder
        '''
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            return
        if hasattr(m, 'weight'):
            for p in m.parameters(): p.requires_grad_(b)

    def freeze(self):
        '''
        freezes the network except for the batch norm layers
        '''
        self.apply(partial(self._set_grad, b=False))

    def unfreeze(self):
        '''
        unfreezes all network
        '''
        self.apply(partial(self._set_grad, b=True))

def xresnet18(c_in=3, c_out=1000):
    '''
    Returns a XResnet with 18 layers
    '''
    return XResNet.create(1, [2, 2, 2, 2], c_in, c_out)

def xresnet34(c_in=3, c_out=1000):
    '''
    Returns a XResnet with 34 layers
    '''
    return XResNet.create(1, [3, 4, 6, 3], c_in, c_out)

def xresnet50(c_in=3, c_out=1000):
    '''
    Returns a XResnet with 50 layers
    '''
    return XResNet.create(4, [3, 4, 6, 3], c_in, c_out)

def xresnet101(c_in=3, c_out=1000):
    '''
    Returns a XResnet with 101 layers
    '''
    return XResNet.create(4, [3, 4, 23, 3], c_in, c_out)

def xresnet152(c_in=3, c_out=1000):
    '''
    Returns a XResnet with 152 layers
    '''
    return XResNet.create(4, [3, 8, 36, 3], c_in, c_out)
