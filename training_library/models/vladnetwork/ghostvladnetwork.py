from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from ...modules.layers import VladPoolingLayer

__all__ = ['VladNetwork']

def make_base_layer(filters, is_bottleneck=False):
    '''
    makes both bottleneck and identity layers
    '''
    _in, _out = filters
    layers_configs = [{'in_channels': _in, 'out_channels': _in, 'kernel_size': 1,
                       'stride': 2, 'padding': 0, 'bias': False}] if is_bottleneck \
                      else [{'in_channels': _out, 'out_channels': _in, 'kernel_size': 1,
                             'stride': 1, 'padding': 0, 'bias': False}]

    layers_configs.extend([{'in_channels': _in, 'out_channels': _in, 'kernel_size': 3,
                            'stride': 1, 'padding': 1, 'bias': False},\
                           {'in_channels': _in, 'out_channels': _out, 'kernel_size': 1,
                            'stride': 1, 'padding': 0, 'bias': False}])
    layers = []
    for config in layers_configs:
        conv = nn.Conv2d(**config)
        b_norm = nn.BatchNorm2d(config['out_channels'])
        act = nn.ReLU(inplace=True)
        layers.extend([conv, b_norm, act])
    return nn.Sequential(*layers[:-1])


class IdentityBlock(nn.Module):
    '''
    the identity layer
    '''
    def __init__(self, n_filters: [int]):
        super(IdentityBlock, self).__init__()
        self.conv = make_base_layer(n_filters, is_bottleneck=False)
    def forward(self, *inputs):
        return self.conv(inputs[0]) + inputs[0]

class BottleneckBlock(nn.Module):
    '''
    layer with bottleneck
    '''
    def __init__(self, n_filters: [int]):
        super(BottleneckBlock, self).__init__()
        self.conv = make_base_layer(n_filters, is_bottleneck=True)
        self.bottleneck = nn.Sequential(nn.Conv2d(*n_filters
                                                  , kernel_size=(1, 1)
                                                  , stride=(2, 2)
                                                  , bias=False),
                                        nn.BatchNorm2d(n_filters[1]))
    def forward(self, *inputs):
        return self.conv(inputs[0]) + self.bottleneck(inputs[0])

def make_thin_layer(depth, layers, is_first_layer=False):
    '''
    makes a thinresnet layer
    '''
    if is_first_layer:
        first_bottleneck = BottleneckBlock(layers)
        first_bottleneck.conv[0] = nn.Conv2d(in_channels=64, out_channels=48,
                                             kernel_size=1, stride=1, padding=0, bias=False)
        first_bottleneck.bottleneck[0] = nn.Conv2d(in_channels=64, out_channels=96,
                                                   kernel_size=1, stride=1, padding=0, bias=False)
    else:
        first_bottleneck = BottleneckBlock(layers)
    thin_layers = [first_bottleneck, nn.ReLU(inplace=True)]
    for _ in range(depth - 1):
        thin_layers.extend([IdentityBlock(layers), nn.ReLU(inplace=True)])
    return nn.Sequential(*thin_layers)


class ThinResnet(nn.Module):
    '''
    thin resnet implemented according to the keras implementation
    '''
    def __init__(self, input_channels=1):
        super(ThinResnet, self).__init__()
        #initial layer
        self.initial = nn.Sequential(nn.Conv2d(input_channels, 64, (7, 7)
                                               , padding=(3, 3), bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        #subsequent layers
        sections_layers = [[96, 128], [128, 256], [256, 512]]
        sections = [make_thin_layer(depth=2, layers=[48, 96], is_first_layer=True)]
        for l_number in sections_layers:
            sections.append(make_thin_layer(depth=3, layers=l_number))
        self.sections = nn.Sequential(*sections)
        self.last_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))

    def forward(self, *inputs):
        output = self.initial(inputs[0])
        for layer in [self.sections, self.last_pool]:
            output = layer(output)
        return output


class VladNetwork(nn.Module):
    '''
    Vlad network
    '''
    def __init__(self, encoder, vlad_pool, bottleneck_dim=512):
        super(VladNetwork, self).__init__()
        self.encoder = encoder
        self.vlad_pool = vlad_pool
        #dimensionanity reduction to bottleneck_dim
        self.reduction = \
        nn.Sequential(OrderedDict([('linear', nn.Linear(vlad_pool.output_dimension,
                                                        bottleneck_dim, bias=True)),
                                   ('activation', nn.ReLU(inplace=True))]))
    def forward(self, *inputs):
        output = self.encoder(inputs[0])
        output = self.vlad_pool(output)
        output = self.reduction(output)
        #output is being normalized
        return F.normalize(input=output, p=2, dim=-1)

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

    @classmethod
    def load_keras_arch(cls, state_dict=None):
        '''
        loads model from file with the archtecture as the model from Keras
        '''
        encoder = ThinResnet()
        vlad_layer = VladPoolingLayer(n_clusters=8, g_clusters=2, d_dim=512)
        network = cls(encoder=encoder, vlad_pool=vlad_layer, bottleneck_dim=512)
        if state_dict:
            checkpoint = torch.load(state_dict)
            network.load_state_dict(checkpoint)
        return network

    @classmethod
    def build_network(cls, encoder, n_clusters=8, g_clusters=2, d_dim=512, output_dim=512):
        '''
        Builds a network with user specified clusters and D dimension
        '''
        vlad_layer = VladPoolingLayer(n_clusters=n_clusters, g_clusters=g_clusters, d_dim=d_dim)
        network = cls(encoder=encoder, vlad_pool=vlad_layer, bottleneck_dim=output_dim)

        return network
