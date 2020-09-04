"""Functions for building the face recognition inference in Torch.
"""

from collections import OrderedDict
import numpy as np

import torch
from torch.nn import Conv2d

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
    h5key = f'{path}/{name}/weights:0'

    weight = h5utils.read(h5file, h5key)
    weight = np.transpose(weight, axes=[3, 2, 0, 1])
    check_shapes(weight.shape, layer.weight.shape, h5key)

    print(h5key, weight.shape, weight.dtype)
    layer.weight = torch.nn.Parameter(torch.from_numpy(weight))

    if layer.bias is not None:
        h5key = '{}/{}/biases:0'.format(path, name)
        bias = h5utils.read(h5file, h5key)
        check_shapes(bias.shape, layer.bias.shape, h5key)

        print(h5key, bias.shape, bias.dtype)
        layer.bias = torch.nn.Parameter(torch.from_numpy(bias))


def initialize_layers(layers, h5file, path):
    for name, layer in layers.items():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            initialize_conv2d(layer, h5file, path, name)


class Block35(torch.nn.Module):
    """Builds the 35x35 resnet block.
    stride=1, padding=SAME
    """

    def __init__(self, h5file, scale=1., idx=None):
        super().__init__()
        self.scale = scale
        in_channels = 256

        # scope Branch_0
        path = f'InceptionResnetV1/Repeat/block35_{idx}/Branch_0'

        layers = OrderedDict({
            'Conv2d_1x1': Conv2d(in_channels, 32, kernel_size=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU()
        })
        initialize_layers(layers, h5file, path)
        self.tower_conv1 = torch.nn.Sequential(layers)

        # scope Branch_1
        path = f'InceptionResnetV1/Repeat/block35_{idx}/Branch_1'

        layers = OrderedDict({
            'Conv2d_0a_1x1': Conv2d(in_channels, 32, kernel_size=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU(),
            'Conv2d_0b_3x3': Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            'relu2': torch.nn.ReLU()
        })
        initialize_layers(layers, h5file, path)
        self.tower_conv2 = torch.nn.Sequential(layers)

        # scope Branch_2
        path = f'InceptionResnetV1/Repeat/block35_{idx}/Branch_2'

        layers = OrderedDict({
            'Conv2d_0a_1x1': Conv2d(in_channels, 32, kernel_size=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU(),
            'Conv2d_0b_3x3': Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            'relu2': torch.nn.ReLU(),
            'Conv2d_0c_3x3': Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            'relu3': torch.nn.ReLU()
        })
        initialize_layers(layers, h5file, path)
        self.tower_conv3 = torch.nn.Sequential(layers)

        # InceptionResnetV1/Repeat/block35_1/Conv2d_1x1/weights:0
        path = f'InceptionResnetV1/Repeat/block35_{idx}'

        conv2d = Conv2d(96, 256, kernel_size=1, padding=0, bias=True)
        initialize_conv2d(conv2d, h5file, path, 'Conv2d_1x1')
        self.conv2d = conv2d

        self.activation_fn = torch.nn.ReLU()

    def forward(self, input_ids, past=None):
        mixed = torch.cat((self.tower_conv1(input_ids),
                           self.tower_conv2(input_ids),
                           self.tower_conv3(input_ids)), dim=1)

        input_ids += self.scale * self.conv2d(mixed)
        input_ids = self.activation_fn(input_ids)

        return input_ids


class Block17(torch.nn.Module):
    """Builds the 17x17 resnet block
    stride=1, padding=SAME
    """

    def __init__(self, h5file, scale=1., idx=None):
        super().__init__()
        self.scale = scale
        in_channels = 896

        # scope Branch_0
        path = f'InceptionResnetV1/Repeat_1/block17_{idx}/Branch_0'

        layers = OrderedDict({
            'Conv2d_1x1': Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU()
        })
        initialize_layers(layers, h5file, path)
        self.tower_conv1 = torch.nn.Sequential(layers)

        # scope Branch_1
        path = f'InceptionResnetV1/Repeat_1/block17_{idx}/Branch_1'

        layers = OrderedDict({
            'Conv2d_0a_1x1': Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU(),
            'Conv2d_0b_1x7': Conv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=False),
            'relu2': torch.nn.ReLU(),
            'Conv2d_0c_7x1': Conv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0), bias=False),
            'relu3': torch.nn.ReLU()
        })
        initialize_layers(layers, h5file, path)
        self.tower_conv2 = torch.nn.Sequential(layers)

        # InceptionResnetV1/Repeat_1/block17_1/Conv2d_1x1
        path = f'InceptionResnetV1/Repeat_1/block17_{idx}'

        conv2d = Conv2d(256, 896, kernel_size=1, padding=0, bias=True)
        initialize_conv2d(conv2d, h5file, path, 'Conv2d_1x1')
        self.conv2d = conv2d

        self.activation_fn = torch.nn.ReLU()

    def forward(self, input_ids, past=None):
        mixed = torch.cat((self.tower_conv1(input_ids),
                           self.tower_conv2(input_ids)), dim=1)

        input_ids += self.scale * self.conv2d(mixed)
        input_ids = self.activation_fn(input_ids)

        return input_ids


class Block8(torch.nn.Module):
    """Builds the 8x8 resnet block.
    stride=1, padding=SAME
    """

    def __init__(self, h5file, scale=1., idx=None):
        super().__init__()
        self.scale = scale
        in_channels = 1792

        # scope Branch_0
        path = f'InceptionResnetV1/Repeat_2/block8_{idx}/Branch_0'

        layers = OrderedDict({
            'Conv2d_1x1': Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU()
        })
        initialize_layers(layers, h5file, path)
        self.tower_conv1 = torch.nn.Sequential(layers)

        # scope Branch_1
        path = f'InceptionResnetV1/Repeat_2/block8_{idx}/Branch_1'

        layers = OrderedDict({
            'Conv2d_0a_1x1': Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU(),
            'Conv2d_0b_1x3': Conv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            'relu2': torch.nn.ReLU(),
            'Conv2d_0c_3x1': Conv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
            'relu3': torch.nn.ReLU()
        })
        initialize_layers(layers, h5file, path)
        self.tower_conv2 = torch.nn.Sequential(layers)

        # InceptionResnetV1/Repeat_1/block17_1/Conv2d_1x1
        path = f'InceptionResnetV1/Repeat_2/block8_{idx}'

        conv2d = Conv2d(384, 1792, kernel_size=1, padding=0, bias=True)
        initialize_conv2d(conv2d, h5file, path, 'Conv2d_1x1')
        self.conv2d = conv2d

        self.activation_fn = torch.nn.ReLU()

    def forward(self, input_ids, past=None):
        mixed = torch.cat((self.tower_conv1(input_ids),
                           self.tower_conv2(input_ids)), dim=1)

        input_ids += self.scale * self.conv2d(mixed)
        input_ids = self.activation_fn(input_ids)

        return input_ids


class ReductionA(torch.nn.Module):
    """
    stride=1, padding=SAME
    """

    def __init__(self, h5file):
        super().__init__()

        # scope Branch_0
        path = 'InceptionResnetV1/Mixed_6a/Branch_0'

        layers = OrderedDict({
            'Conv2d_1a_3x3': Conv2d(256, 384, kernel_size=3, stride=2, padding=0, bias=False),
            'relu1': torch.nn.ReLU()
        })
        initialize_layers(layers, h5file, path)
        self.tower_conv1 = torch.nn.Sequential(layers)

        # scope Branch_1
        path = 'InceptionResnetV1/Mixed_6a/Branch_1'

        layers = OrderedDict({
            'Conv2d_0a_1x1': Conv2d(256, 192, kernel_size=1, stride=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU(),
            'Conv2d_0b_3x3': Conv2d(192, 192, kernel_size=3,  stride=1, padding=1, bias=False),
            'relu2': torch.nn.ReLU(),
            'Conv2d_1a_3x3': Conv2d(192, 256, kernel_size=3,  stride=2, padding=0, bias=False),
            'relu3': torch.nn.ReLU()
        })
        initialize_layers(layers, h5file, path)
        self.tower_conv2 = torch.nn.Sequential(layers)

        self.max_pool = torch.nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, input_ids, past=None):
        input_ids = torch.cat([self.tower_conv1(input_ids),
                               self.tower_conv2(input_ids),
                               self.max_pool(input_ids)], dim=1)

        return input_ids


class ReductionB(torch.nn.Module):
    """
    stride=1, padding=SAME
    """

    def __init__(self, h5file):
        super().__init__()
        in_channels = 896

        # scope Branch_0
        path = 'InceptionResnetV1/Mixed_7a/Branch_0'

        layers = OrderedDict({
            'Conv2d_0a_1x1': Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU(),
            'Conv2d_1a_3x3': Conv2d(256, 384, kernel_size=3, stride=2, padding=0, bias=False),
            'relu2': torch.nn.ReLU()
        })
        initialize_layers(layers, h5file, path)
        self.tower_conv1 = torch.nn.Sequential(layers)

        # scope Branch_1
        path = 'InceptionResnetV1/Mixed_7a/Branch_1'

        layers = OrderedDict({
            'Conv2d_0a_1x1': Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU(),
            'Conv2d_1a_3x3': Conv2d(256, 256, kernel_size=3,  stride=2, padding=0, bias=False),
            'relu2': torch.nn.ReLU()
        })
        initialize_layers(layers, h5file, path)
        self.tower_conv2 = torch.nn.Sequential(layers)

        # scope Branch_2
        path = 'InceptionResnetV1/Mixed_7a/Branch_2'

        layers = OrderedDict({
            'Conv2d_0a_1x1': Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU(),
            'Conv2d_0b_3x3': Conv2d(256, 256, kernel_size=3,  stride=1, padding=1, bias=False),
            'relu2': torch.nn.ReLU(),
            'Conv2d_1a_3x3': Conv2d(256, 256, kernel_size=3, stride=2, padding=0, bias=False),
            'relu3': torch.nn.ReLU()
        })
        initialize_layers(layers, h5file, path)
        self.tower_conv3 = torch.nn.Sequential(layers)

        self.max_pool = torch.nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, input_ids, past=None):
        input_ids = torch.cat([self.tower_conv1(input_ids),
                               self.tower_conv2(input_ids),
                               self.tower_conv3(input_ids),
                               self.max_pool(input_ids)], dim=1)

        return input_ids


class FaceNet(torch.nn.Module):
    def __init__(self, h5file):
        super().__init__()
        self.h5file = h5file

        layers = OrderedDict({
            'Conv2d_1a_3x3': torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0, bias=False),
            'relu1': torch.nn.ReLU(),
            'Conv2d_2a_3x3': torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False),
            'relu2': torch.nn.ReLU(),
            'Conv2d_2b_3x3': torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            'relu3': torch.nn.ReLU(),
            'pool': torch.nn.MaxPool2d(3, stride=2, padding=0),
            'Conv2d_3b_1x1': torch.nn.Conv2d(64, 80, kernel_size=1, stride=1, padding=0, bias=False),
            'relu4': torch.nn.ReLU(),
            'Conv2d_4a_3x3': torch.nn.Conv2d(80, 192, kernel_size=3, stride=1, padding=0, bias=False),
            'relu5': torch.nn.ReLU(),
            'Conv2d_4b_3x3': torch.nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=0, bias=False),
            'relu6': torch.nn.ReLU()
        })

        initialize_layers(layers, h5file, 'InceptionResnetV1')
        self.sequential = torch.nn.Sequential(layers)

        layers = OrderedDict()
        for idx in range(5):
            layers[f'block35_{idx+1}'] = Block35(h5file, scale=0.17, idx=idx+1)
        self.block35 = torch.nn.Sequential(layers)

        self.reduction_a = ReductionA(h5file)

        # 10 x Inception-Resnet-B
        layers = OrderedDict()
        for idx in range(10):
            layers[f'block17_{idx+1}'] = Block17(h5file, scale=0.10, idx=idx+1)
        self.block17 = torch.nn.Sequential(layers)

        # Reduction-B
        self.reduction_b = ReductionB(h5file)

        # 5 x Inception-Resnet-C
        layers = OrderedDict()
        for idx in range(5):
            layers[f'block8_{idx+1}'] = Block8(h5file, scale=0.20, idx=idx+1)
        self.block8 = torch.nn.Sequential(layers)

    def forward(self, input_ids, past=None):
        input_ids = self.sequential.forward(input_ids)
        input_ids = self.block35.forward(input_ids)
        input_ids = self.reduction_a.forward(input_ids)
        input_ids = self.block17.forward(input_ids)
        input_ids = self.reduction_b.forward(input_ids)
        input_ids = self.block8.forward(input_ids)

        return input_ids

