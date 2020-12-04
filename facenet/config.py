# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

import sys
import yaml
from pathlib import Path
from datetime import datetime
import importlib
import numpy as np
import random
import tensorflow as tf

from facenet import ioutils

src_dir = Path(__file__).parents[1]
default_model = src_dir.joinpath('models', '20201008-183421')
default_batch_size = 64
image_size = 160

data_dir = Path(__file__).parents[1].joinpath('data')
faces_dir = data_dir.joinpath('faces')


file_extension = '.png'


def subdir():
    return datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')


def default_app_config(apps_file_name):
    config_dir = Path(Path(apps_file_name).parent).joinpath('configs')
    config_name = Path(apps_file_name).stem
    return config_dir.joinpath(config_name + '.yaml')


class YAMLConfig:
    """Object representing YAML settings as a dict-like object with values as fields
    """

    def __init__(self, item):
        if isinstance(item, dict):
            self.update_from_dict(item)
        else:
            self.update_from_file(item)

    def __repr__(self):
        shift = 3 * ' '

        def get_str(obj, ident=''):
            s = ''
            for key, item in obj.items():
                if isinstance(item, YAMLConfig):
                    s += '{}{}: \n{}'.format(ident, key, get_str(item, ident=ident + shift))
                else:
                    s += '{}{}: {}\n'.format(ident, key, str(item))
            return s

        return get_str(self)

    def __getattr__(self, name):
        return self.__dict__.get(name, YAMLConfig({}))

    def __bool__(self):
        return bool(self.__dict__)

    @staticmethod
    def check_item(s):
        if not isinstance(s, str):
            return s
        if s.lower() == 'none':
            return None
        if s.lower() == 'false':
            return False
        if s.lower() == 'true':
            return True
        return s

    def update_from_dict(self, dct):
        """Update config from dict

        :param dct: dict
        """
        for key, item in dct.items():
            if isinstance(item, dict):
                setattr(self, key, YAMLConfig(item))
            else:
                setattr(self, key, self.check_item(item))

    def update_from_file(self, path):
        """Update config from YAML file
        """
        if not path.exists():
            raise ValueError('file {} does not exist'.format(path))

        with path.open('r') as f:
            self.update_from_dict(yaml.safe_load(f.read()))

    def items(self):
        return self.__dict__.items()

    def exists(self, name):
        return True if name in self.__dict__.keys() else False


class Embeddings(YAMLConfig):
    def __init__(self, args_):
        YAMLConfig.__init__(self, args_['config'])
        if not self.model.path:
            self.model.path = default_model

        # if not self.output:
        suffix = self.output
        if suffix[0] != '.':
            suffix = '.' + suffix

        self.output = Path(self.dataset.path + self.model.path.stem).with_suffix(suffix)
        self.output = Path(self.output).expanduser()

        if self.output.suffix not in ['.h5', '.tfrecord']:
            raise ValueError('Invalid suffix for output file, must either be h5 or tfrecord.')

        self.log_dir = self.output.parent
        self.log_file = self.output.with_suffix('.txt')

        if not self.batch_size:
            self.batch_size = default_batch_size

        # write arguments and store some git revision info in a text files in the log directory
        ioutils.write_arguments(self, Path(self.log_dir, self.output.stem + '_arguments.yaml'))
        ioutils.store_revision_info(self.log_dir, sys.argv)


class TrainClassifier(YAMLConfig):
    def __init__(self, config):
        YAMLConfig.__init__(self, config['config'])

        if not self.seed:
            self.seed = 0

        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.classifier.path = Path(self.classifier.path).expanduser() / subdir()

        if not self.model.path:
            self.model.path = default_model

        if not self.batch_size:
            self.batch_size = default_batch_size

        self.log_dir = self.classifier.path / 'logs'
        self.log_file = self.log_dir / 'report.txt'

        # write arguments and store some git revision info in a text files in the log directory
        ioutils.write_arguments(self, self.log_dir.joinpath('arguments.yaml'))
        ioutils.store_revision_info(self.log_dir, sys.argv)


class Validate(YAMLConfig):
    def __init__(self, args_):
        YAMLConfig.__init__(self, args_['config'])
        if not self.seed:
            self.seed = 0
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        if not self.model.path:
            self.model.path = default_model

        if not self.file:
            self.file = Path(self.model.path).expanduser().joinpath('report.txt')
        else:
            self.file = Path(self.file).expanduser()

        if not self.batch_size:
            self.batch_size = default_batch_size

        if not self.dataset.min_nrof_images:
            self.dataset.min_nrof_images = 1

        # write arguments and store some git revision info in a text files in the log directory
        ioutils.write_arguments(self, self.logs.joinpath('arguments.yaml'))
        ioutils.store_revision_info(self.logs, sys.argv)


class TrainOptions(YAMLConfig):
    def __init__(self, args_, subdir=None):
        YAMLConfig.__init__(self, args_['config'])

        if not self.seed:
            self.seed = 0
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        if not self.batch_size:
            self.batch_size = default_batch_size

        if subdir is None:
            self.model.path = Path(self.model.path).expanduser()
        else:
            self.model.path = Path(self.model.path).expanduser().joinpath(subdir)

        if not self.dataset.min_nrof_images:
            self.dataset.min_nrof_images = 1

        if not self.validate.dataset.min_nrof_images:
            self.validate.dataset.min_nrof_images = 1

        self.logs = self.model.path.joinpath('logs')
        self.h5file = self.logs.joinpath('report.h5')
        self.txtfile = self.logs.joinpath('report.txt')

        if self.model.config is None:
            network = importlib.import_module(self.model.module)
            self.model.update_from_file(network.config_file)

        # learning rate options
        if self.train.learning_rate.schedule:
            self.train.epoch.nrof_epochs = self.train.learning_rate.schedule[-1][0]

        if self.validate:
            self.validate.batch_size = self.batch_size
            self.validate.image.size = self.image.size
            self.validate.image.standardization = self.image.standardization

        if not self.validate.file:
            self.validate.file = Path(self.model.path).expanduser().joinpath('report.txt')

        # write arguments and store some git revision info in a text files in the log directory
        ioutils.write_arguments(self, self.logs.joinpath('arguments.yaml'))
        ioutils.store_revision_info(self.logs, sys.argv)
