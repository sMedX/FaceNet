"""Functions for building the face recognition inference in Torch.
"""

from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from facenet import h5utils


def image_processing(image, eps=1e-3):
    image = torch.from_numpy(image)
    image = image.float()

    min_value = torch.min(image)
    max_value = torch.max(image)
    dynamic_range = torch.max(max_value - min_value, torch.tensor(eps))
    image = (2*image - (max_value + min_value))/dynamic_range

    return image


def check_shapes(shape1, shape2, key):
    if shape1 != shape2:
        raise ValueError(f'Shapes {tuple(shape1)} and {tuple(shape2)} do not match for key {key}')


def initialize_conv2d(layer, h5file, path, name):
    h5key = f'{path}/{name}/weights'

    weight = h5utils.read(h5file, h5key)
    weight = np.transpose(weight, axes=[3, 2, 0, 1])
    check_shapes(weight.shape, layer.weight.shape, h5key)

    print(h5key, weight.shape, weight.dtype)
    layer.weight = nn.Parameter(torch.from_numpy(weight))

    if layer.bias is not None:
        h5key = f'{path}/{name}/biases'
        bias = h5utils.read(h5file, h5key)
        check_shapes(bias.shape, layer.bias.shape, h5key)

        print(h5key, bias.shape, bias.dtype)
        layer.bias = nn.Parameter(torch.from_numpy(bias))


def initialize_linear(layer, h5file, path, name):
    h5key = f'{path}/{name}/weights'

    weight = h5utils.read(h5file, h5key)
    weight = np.transpose(weight)
    check_shapes(weight.shape, layer.weight.shape, h5key)

    print(h5key, weight.shape, weight.dtype)
    layer.weight = nn.Parameter(torch.from_numpy(weight))

    if layer.bias is not None:
        h5key = f'{path}/{name}/biases'
        bias = h5utils.read(h5file, h5key)
        check_shapes(bias.shape, layer.bias.shape, h5key)

        print(h5key, bias.shape, bias.dtype)
        layer.bias = nn.Parameter(torch.from_numpy(bias))


def initialize_layers(layers, h5file, path):
    for name, layer in layers.items():
        if isinstance(layer, nn.Conv2d):
            initialize_conv2d(layer, h5file, path, name)
        if isinstance(layer, nn.Linear):
            initialize_linear(layer, h5file, path, name)


class Block35(nn.Module):
    """Builds the 35x35 resnet block.
    stride=1, padding=SAME
    """

    def __init__(self, h5file, path, scale=1.):
        super().__init__()
        self.scale = scale
        in_channels = 256

        # scope Branch_0
        layers = OrderedDict({
            'Conv2d_1x1': nn.Conv2d(in_channels, 32, kernel_size=1, padding=0, bias=True),
            'relu1': nn.ReLU()
        })
        initialize_layers(layers, h5file, path + '/Branch_0')
        self.tower_conv1 = nn.Sequential(layers)

        # scope Branch_1
        layers = OrderedDict({
            'Conv2d_0a_1x1': nn.Conv2d(in_channels, 32, kernel_size=1, padding=0, bias=True),
            'relu1': nn.ReLU(),
            'Conv2d_0b_3x3': nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            'relu2': nn.ReLU()
        })
        initialize_layers(layers, h5file, path + '/Branch_1')
        self.tower_conv2 = nn.Sequential(layers)

        # scope Branch_2
        layers = OrderedDict({
            'Conv2d_0a_1x1': nn.Conv2d(in_channels, 32, kernel_size=1, padding=0, bias=True),
            'relu1': nn.ReLU(),
            'Conv2d_0b_3x3': nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            'relu2': nn.ReLU(),
            'Conv2d_0c_3x3': nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            'relu3': nn.ReLU()
        })
        initialize_layers(layers, h5file, path + '/Branch_2')
        self.tower_conv3 = nn.Sequential(layers)

        # InceptionResnetV1/Repeat/block35_1/Conv2d_1x1/weights:0
        conv2d = nn.Conv2d(96, 256, kernel_size=1, padding=0, bias=True)
        initialize_conv2d(conv2d, h5file, path, 'Conv2d_1x1')
        self.conv2d = conv2d

    def forward(self, input_ids, past=None):
        mixed = torch.cat((self.tower_conv1(input_ids),
                           self.tower_conv2(input_ids),
                           self.tower_conv3(input_ids)), dim=1)

        input_ids += self.scale * self.conv2d(mixed)

        return input_ids


class Block17(nn.Module):
    """Builds the 17x17 resnet block
    stride=1, padding=SAME
    """

    def __init__(self, h5file, path, scale=1.0):
        super().__init__()
        self.scale = scale
        in_channels = 896

        # scope Branch_0
        layers = OrderedDict({
            'Conv2d_1x1': nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0, bias=True),
            'relu1': nn.ReLU()
        })
        initialize_layers(layers, h5file, path + '/Branch_0')
        self.tower_conv1 = nn.Sequential(layers)

        # scope Branch_1
        layers = OrderedDict({
            'Conv2d_0a_1x1': nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0, bias=True),
            'relu1': nn.ReLU(),
            'Conv2d_0b_1x7': nn.Conv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
            'relu2': nn.ReLU(),
            'Conv2d_0c_7x1': nn.Conv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0), bias=True),
            'relu3': nn.ReLU()
        })
        initialize_layers(layers, h5file, path + '/Branch_1')
        self.tower_conv2 = nn.Sequential(layers)

        # InceptionResnetV1/Repeat_1/block17_1/Conv2d_1x1
        conv2d = nn.Conv2d(256, 896, kernel_size=1, padding=0, bias=True)
        initialize_conv2d(conv2d, h5file, path, 'Conv2d_1x1')
        self.conv2d = conv2d

    def forward(self, input_ids, past=None):
        mixed = torch.cat((self.tower_conv1(input_ids),
                           self.tower_conv2(input_ids)), dim=1)

        input_ids += self.scale * self.conv2d(mixed)

        return input_ids


class Block8(nn.Module):
    """Builds the 8x8 resnet block.
    stride=1, padding=SAME
    """

    def __init__(self, h5file, path, scale=1.0):
        super().__init__()
        self.scale = scale
        in_channels = 1792

        # scope Branch_0
        layers = OrderedDict({
            'Conv2d_1x1': nn.Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0, bias=True),
            'relu1': nn.ReLU()
        })
        initialize_layers(layers, h5file, f'{path}/Branch_0')
        self.tower_conv1 = nn.Sequential(layers)

        # scope Branch_1
        layers = OrderedDict({
            'Conv2d_0a_1x1': nn.Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0, bias=True),
            'relu1': nn.ReLU(),
            'Conv2d_0b_1x3': nn.Conv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            'relu2': nn.ReLU(),
            'Conv2d_0c_3x1': nn.Conv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            'relu3': nn.ReLU()
        })
        initialize_layers(layers, h5file, f'{path}/Branch_1')
        self.tower_conv2 = nn.Sequential(layers)

        # InceptionResnetV1/Repeat_1/block17_1/Conv2d_1x1
        conv2d = nn.Conv2d(384, 1792, kernel_size=1, padding=0, bias=True)
        initialize_conv2d(conv2d, h5file, path, 'Conv2d_1x1')
        self.conv2d = conv2d

    def forward(self, input_ids, past=None):
        mixed = torch.cat((self.tower_conv1(input_ids),
                           self.tower_conv2(input_ids)), dim=1)

        input_ids += self.scale * self.conv2d(mixed)

        return input_ids


class ReductionA(nn.Module):
    """
    stride=1, padding=SAME
    """

    def __init__(self, h5file, path):
        super().__init__()

        # scope Branch_0
        layers = OrderedDict({
            'Conv2d_1a_3x3': nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=0, bias=True),
            'relu1': nn.ReLU()
        })
        initialize_layers(layers, h5file, path + '/Branch_0')
        self.tower_conv1 = nn.Sequential(layers)

        # scope Branch_1
        layers = OrderedDict({
            'Conv2d_0a_1x1': nn.Conv2d(256, 192, kernel_size=1, stride=1, padding=0, bias=True),
            'relu1': nn.ReLU(),
            'Conv2d_0b_3x3': nn.Conv2d(192, 192, kernel_size=3,  stride=1, padding=1, bias=True),
            'relu2': nn.ReLU(),
            'Conv2d_1a_3x3': nn.Conv2d(192, 256, kernel_size=3,  stride=2, padding=0, bias=True),
            'relu3': nn.ReLU()
        })
        initialize_layers(layers, h5file,  path + '/Branch_1')
        self.tower_conv2 = nn.Sequential(layers)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, input_ids, past=None):
        input_ids = torch.cat([self.tower_conv1(input_ids),
                               self.tower_conv2(input_ids),
                               self.max_pool(input_ids)], dim=1)

        return input_ids


class ReductionB(nn.Module):
    """
    stride=1, padding=SAME
    """

    def __init__(self, h5file, path):
        super().__init__()
        in_channels = 896

        # scope Branch_0
        layers = OrderedDict({
            'Conv2d_0a_1x1': nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=True),
            'relu1': nn.ReLU(),
            'Conv2d_1a_3x3': nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=0, bias=True),
            'relu2': nn.ReLU()
        })
        initialize_layers(layers, h5file, path + '/Branch_0')
        self.tower_conv1 = nn.Sequential(layers)

        # scope Branch_1
        layers = OrderedDict({
            'Conv2d_0a_1x1': nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=True),
            'relu1': nn.ReLU(),
            'Conv2d_1a_3x3': nn.Conv2d(256, 256, kernel_size=3,  stride=2, padding=0, bias=True),
            'relu2': nn.ReLU()
        })
        initialize_layers(layers, h5file, path + '/Branch_1')
        self.tower_conv2 = nn.Sequential(layers)

        # scope Branch_2
        layers = OrderedDict({
            'Conv2d_0a_1x1': nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=True),
            'relu1': nn.ReLU(),
            'Conv2d_0b_3x3': nn.Conv2d(256, 256, kernel_size=3,  stride=1, padding=1, bias=True),
            'relu2': nn.ReLU(),
            'Conv2d_1a_3x3': nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, bias=True),
            'relu3': nn.ReLU()
        })
        initialize_layers(layers, h5file, path + '/Branch_2')
        self.tower_conv3 = nn.Sequential(layers)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, input_ids, past=None):
        input_ids = torch.cat([self.tower_conv1(input_ids),
                               self.tower_conv2(input_ids),
                               self.tower_conv3(input_ids),
                               self.max_pool(input_ids)], dim=1)

        return input_ids


class FaceNet(nn.Module):
    def __init__(self, h5file):
        super().__init__()
        self.h5file = h5file

        layers = OrderedDict({
            'Conv2d_1a_3x3': nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0, bias=True),
            'Conv2d_1a_3x3/Relu': nn.ReLU(inplace=True),
            'Conv2d_2a_3x3': nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=True),
            'Conv2d_2a_3x3/Relu': nn.ReLU(inplace=True),
            'Conv2d_2b_3x3': nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            'Conv2d_2b_3x3/Relu': nn.ReLU(inplace=True),
            'MaxPool_3a_3x3': nn.MaxPool2d(3, stride=2, padding=0),
            'Conv2d_3b_1x1': nn.Conv2d(64, 80, kernel_size=1, stride=1, padding=0, bias=True),
            'Conv2d_3b_1x1/Relu': nn.ReLU(inplace=True),
            'Conv2d_4a_3x3': nn.Conv2d(80, 192, kernel_size=3, stride=1, padding=0, bias=True),
            'Conv2d_4a_3x3/Relu': nn.ReLU(inplace=True),
            'Conv2d_4b_3x3': nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=0, bias=True),
            'Conv2d_4b_3x3/Relu': nn.ReLU(inplace=True)
        })

        initialize_layers(layers, h5file, 'InceptionResnetV1')
        self.sequential = nn.Sequential(layers)

        layers = OrderedDict()
        for idx in range(5):
            path = f'InceptionResnetV1/Repeat/block35_{idx+1}'
            layers[f'block35_{idx+1}'] = Block35(h5file, path, scale=0.17)
            layers[f'block35_{idx+1}_relu'] = nn.ReLU()
        self.block35 = nn.Sequential(layers)

        self.reduction_a = ReductionA(h5file, 'InceptionResnetV1/Mixed_6a')

        # 10 x Inception-Resnet-B
        layers = OrderedDict()
        for idx in range(10):
            path = f'InceptionResnetV1/Repeat_1/block17_{idx+1}'
            layers[f'block17_{idx+1}'] = Block17(h5file, path, scale=0.10)
            layers[f'block17_{idx+1}_relu'] = nn.ReLU()
        self.block17 = nn.Sequential(layers)

        # Reduction-B
        self.reduction_b = ReductionB(h5file, 'InceptionResnetV1/Mixed_7a')

        # 5 x Inception-Resnet-C
        layers = OrderedDict()
        for idx in range(5):
            path = f'InceptionResnetV1/Repeat_2/block8_{idx+1}'
            layers[f'block8_{idx+1}'] = Block8(h5file, path=path, scale=0.20)
            layers[f'block8_{idx+1}_relu'] = nn.ReLU()

        path = 'InceptionResnetV1/Block8'
        layers[f'block8_{5}'] = Block8(h5file, path, scale=1.0)
        self.block8 = nn.Sequential(layers)

        self.avg_pool2d = nn.AvgPool2d(3, stride=1, padding=0)
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(1792, 512, bias=False)
        initialize_linear(self.linear, h5file, 'InceptionResnetV1', 'Bottleneck')

    def forward(self, input_ids, past=None):
        input_ids = self.sequential.forward(input_ids)
        input_ids = self.block35.forward(input_ids)
        input_ids = self.reduction_a.forward(input_ids)
        input_ids = self.block17.forward(input_ids)
        input_ids = self.reduction_b.forward(input_ids)
        input_ids = self.block8.forward(input_ids)

        input_ids = self.avg_pool2d.forward(input_ids)
        input_ids = self.flatten.forward(input_ids)

        input_ids = self.linear.forward(input_ids)

        return input_ids

