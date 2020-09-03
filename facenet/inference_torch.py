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


def initialize_layers(layers, h5file, path):
    for name, layer in layers.items():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            h5key = '{}/{}/weights:0'.format(path, name)

            weight = h5utils.read(h5file, h5key)
            print(h5key, weight.shape, weight.dtype)
            weight = np.transpose(weight, axes=[3, 2, 0, 1])
            layer.weight = torch.nn.Parameter(torch.from_numpy(weight))

            if layer.bias is not None:
                h5key = '{}/{}/biases:0'.format(path, name)
                bias = h5utils.read(h5file, h5key)
                print(h5key, bias.shape, bias.dtype)
                layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

    return layers


class Block35(torch.nn.Module):
    """Builds the 35x35 resnet block.
    stride=1, padding=SAME
    """

    def __init__(self, h5file, scale=1., idx=None):
        super().__init__()
        self.scale = scale
        in_channels = 256

        # scope Branch_0
        path = 'InceptionResnetV1/Repeat/block35_{}/Branch_0'.format(idx)

        layers = OrderedDict({
            'Conv2d_1x1': Conv2d(in_channels, 32, kernel_size=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU()
        })
        self.tower_conv1 = torch.nn.Sequential(initialize_layers(layers, h5file, path))

        # scope Branch_1
        path = 'InceptionResnetV1/Repeat/block35_{}/Branch_1'.format(idx)

        layers = OrderedDict({
            'Conv2d_0a_1x1': Conv2d(in_channels, 32, kernel_size=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU(),
            'Conv2d_0b_3x3': Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            'relu2': torch.nn.ReLU()
        })
        self.tower_conv2 = torch.nn.Sequential(initialize_layers(layers, h5file, path))

        # scope Branch_2
        path = 'InceptionResnetV1/Repeat/block35_{}/Branch_2'.format(idx)

        layers = OrderedDict({
            'Conv2d_0a_1x1': Conv2d(in_channels, 32, kernel_size=1, padding=0, bias=False),
            'relu1': torch.nn.ReLU(),
            'Conv2d_0b_3x3': Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            'relu2': torch.nn.ReLU(),
            'Conv2d_0c_3x3': Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            'relu3': torch.nn.ReLU()
        })
        self.tower_conv3 = torch.nn.Sequential(initialize_layers(layers, h5file, path))

        # InceptionResnetV1/Repeat/block35_1/Conv2d_1x1/weights:0
        path = 'InceptionResnetV1/Repeat/block35_{}'.format(idx)

        layers = OrderedDict({
            'Conv2d_1x1':  Conv2d(32, 32, kernel_size=1, padding=0, bias=True)
        })
        self.conv2d = torch.nn.Sequential(initialize_layers(layers, h5file, path))

        self.activation_fn = torch.nn.ReLU()

    def forward(self, input_ids, past=None):
        mixed = torch.cat((self.tower_conv1(input_ids),
                           self.tower_conv2(input_ids),
                           self.tower_conv3(input_ids)), dim=1)

        input_ids += self.scale * self.conv2d(mixed)
        input_ids = self.activation_fn(input_ids)

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

        self.sequential = torch.nn.Sequential(initialize_layers(layers, h5file, 'InceptionResnetV1'))

        layers = OrderedDict()
        for idx in range(5):
            layers[f'block35_{idx+1}'] = Block35(h5file, scale=0.17, idx=idx+1)
        self.block35 = torch.nn.Sequential(layers)

        # self.apply(self.init_weights)

    def init_weights(self):
        raise NotImplementedError

    def forward(self, input_ids, past=None):
        out = self.sequential.forward(input_ids)
        out = self.block35.forward(out)
        return out

