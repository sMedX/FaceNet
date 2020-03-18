# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

import sys
import oyaml as yaml
from pathlib import Path
from datetime import datetime
import importlib
import numpy as np
import random
from facenet import ioutils, facenet

src_dir = Path(__file__).parents[1]
file_extension = '.png'


def subdir():
    return datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')


def default_app_config(apps_file_name):
    config_dir = Path(Path(apps_file_name).parent).joinpath('configs')
    config_name = Path(apps_file_name).stem
    return config_dir.joinpath(config_name + '.yaml')


def replace_str_value(x):
    dct = {'none': None, 'false': False, 'true': True}

    if isinstance(x, str):
        for name, value in dct.items():
            if x.lower() == name:
                return value
    return x


# class Namespace:
#     """Simple object for storing attributes.
#     Implements equality by attribute names and values, and provides a simple string representation.
#     """
#
#     def __init__(self, dct):
#         for key, item in dct.items():
#             if isinstance(item, dict):
#                 setattr(self, key, Namespace(item))
#             else:
#                 setattr(self, key, replace_str_value(item))
#
#     def items(self):
#         return self.__dict__.items()
#
#     def __repr__(self):
#         return "<namespace object>"


class YAMLConfig:
    """Object representing YAML settings as a dict-like object with values as fields
    """

    def __init__(self, item):
        if isinstance(item, dict):
            self.update_from_dict(item)
        else:
            self.update_from_file(item)

    def update_from_dict(self, dct):
        """Update config from dict

        :param dct: dict
        """
        for key, item in dct.items():
            if isinstance(item, dict):
                setattr(self, key, YAMLConfig(item))
            else:
                setattr(self, key, replace_str_value(item))

    def update_from_file(self, path):
        """Update config from YAML file
        """
        if path is not None:
            with open(str(path), 'r') as custom_config:
                self.update_from_dict(yaml.safe_load(custom_config.read()))

    def items(self):
        return self.__dict__.items()

    def set_to(self, name, value=None):
        if name not in self.__dict__.keys():
            setattr(self, name, value)

    def __repr__(self):
        return "<config object>"


class TrainOptions(YAMLConfig):
    def __init__(self, args_, subdir=None):
        YAMLConfig.__init__(self, args_['config'])

        if subdir is None:
            self.model.path = Path(self.model.path).expanduser()
        else:
            self.model.path = Path(self.model.path).expanduser().joinpath(subdir)
        self.model.logs = self.model.path.joinpath('logs')

        if self.model.config is None:
            network = importlib.import_module(self.model.module)
            self.model.update_from_file(network.config_file)

        # learning rate options
        if args_['learning_rate'] is not None:
            self.train.learning_rate.value = args_['learning_rate']
        self.train.epoch.max_nrof_epochs = facenet.max_nrof_epochs(self.train.learning_rate)

        np.random.seed(seed=self.seed)
        random.seed(self.seed)

        self.set_to('validation', value=None)
        if self.validation is not None:
            self.validation.batch_size = self.batch_size
            self.validation.image.size = self.image.size
            self.validation.image.standardization = self.image.standardization
            self.validation.validation.file = None

        # write arguments and store some git revision info in a text files in the log directory
        ioutils.write_arguments(self, self.model.logs.joinpath('arguments.yaml'))
        ioutils.store_revision_info(self.model.logs, sys.argv)


class DefaultConfig:
    def __init__(self):
        self.model = src_dir.joinpath('models', '20190822-033520')
        self.pretrained_checkpoint = src_dir.joinpath('models', '20190822-033520', 'model-20190822-033520.ckpt-275')

        # image size (height, width) in pixels
        self.image_size = 160
