# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

import yaml
from pathlib import Path
from datetime import datetime
import importlib
import random

import omegaconf
from omegaconf import OmegaConf

import numpy as np
import tensorflow as tf

from facenet import ioutils

# src_dir = Path(__file__).parents[1]

# directory for default configs
default_config_dir = Path(__file__).parents[0].joinpath('apps', 'configs')
default_config = default_config_dir.joinpath('config.yaml')

# directory for user's configs
user_config_dir = Path(__file__).parents[1].joinpath('configs')
user_config = user_config_dir.joinpath('config.yaml')

# default_dataset = Path('~/datasets/vggface2/train')
#
# default_train_dataset = Path('~/datasets/vggface2/train_extracted_160')
# default_test_dataset = Path('~/datasets/vggface2/test_extracted_160')
#
# default_model = src_dir.joinpath('models', '20201008-183421')
# default_batch_size = 100

# image_margin = 0
# image_size = 160
# image_normalization = 0

# data_dir = Path(__file__).parents[1].joinpath('data')
# faces_dir = data_dir.joinpath('faces')
#
# file_extension = '.png'


def subdir():
    return datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')


def config_paths(app_file_name, custom_config_file):
    config_name = Path(app_file_name).stem + '.yaml'

    paths = [
        default_config,
        default_config_dir.joinpath(config_name),
        user_config,
        user_config_dir.joinpath(config_name)
    ]

    if custom_config_file is not None:
        paths.append(custom_config_file)

    return tuple(paths)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


class Config:
    """Object representing YAML settings as a dict-like object with values as fields
    """

    def __init__(self, dct):
        """Update config from dict
        :param dct: input object
        """
        for key, item in dct.items():
            if isinstance(item, (omegaconf.dictconfig.DictConfig, dict)):
                setattr(self, key, Config(item))
            else:
                setattr(self, key, item)

    def __repr__(self):
        shift = 3 * ' '

        def get_str(obj, ident=''):
            s = ''
            for key, item in obj.items():
                if isinstance(item, Config):
                    s += f'{ident}{key}: \n{get_str(item, ident=ident + shift)}'
                else:
                    s += f'{ident}{key}: {str(item)}\n'
            return s

        return get_str(self)

    def __getattr__(self, name):
        return self.__dict__.get(name, Config({}))

    def __bool__(self):
        return bool(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def exists(self, name):
        return True if name in self.__dict__.keys() else False


class LoadConfigError(Exception):
    pass


def load_config(app_file_name, options):
    """Load configuration from the set of config files
    :param app_file_name
    :param options: Optional path to the custom config file
    :return: The validated config in Config model instance
    """

    paths = config_paths(app_file_name, options['config'])

    cfg = OmegaConf.create()
    new_cfg = None

    for config_path in paths:
        if not config_path.is_file():
            continue

        try:
            new_cfg = OmegaConf.load(config_path)
            cfg = OmegaConf.merge(cfg, new_cfg)
        except Exception as err:
            raise LoadConfigError(f"Cannot load configuration from '{config_path}'\n{err}")

    if new_cfg is None:
        raise LoadConfigError("The configuration has not been loaded.")

    cfg = Config(cfg)

    return cfg


def extract_faces(app_file_name, options):
    cfg = load_config(app_file_name, options)

    if not cfg.outdir:
        cfg.outdir = f'{Path(cfg.dataset.path)}_extracted_{cfg.image.size}'

    cfg.outdir = Path(cfg.outdir).expanduser()
    cfg.logdir = cfg.outdir
    cfg.logfile = cfg.outdir / 'log.txt'
    cfg.h5file = cfg.outdir / 'statistics.h5'

    # set seed for random number generators
    set_seed(cfg.seed)

    # write arguments and store some git revision info in a text files in the log directory
    ioutils.write_arguments(cfg, cfg.logdir.joinpath(Path(app_file_name).stem + '.yaml'))
    ioutils.store_revision_info(cfg.logdir)

    return cfg


def train_softmax(app_file_name, options):
    cfg = load_config(app_file_name, options)

    cfg.model.path = Path(cfg.model.path).expanduser().joinpath(subdir())

    cfg.logdir = cfg.model.path.joinpath('logs')
    cfg.logfile = cfg.logdir.joinpath('statistics.txt')
    cfg.h5file = cfg.logdir.joinpath('statistics.h5')

    if not cfg.model.config:
        network = importlib.import_module(cfg.model.module)
        cfg.model.config = network.default_model_config

    if not cfg.train.epoch.nrof_epochs:
        cfg.train.epoch.nrof_epochs = cfg.train.learning_rate.schedule[-1][0]

    if cfg.validate:
        cfg.validate.batch_size = cfg.batch_size
        cfg.validate.image.size = cfg.image.size
        cfg.validate.image.standardization = cfg.image.standardization

    # set seed for random number generators
    set_seed(cfg.seed)

    # write arguments and store some git revision info in a text files in the log directory
    ioutils.write_arguments(cfg, cfg.logdir.joinpath(Path(app_file_name).stem + '.yaml'))
    ioutils.store_revision_info(cfg.logdir)

    return cfg


def embeddings(app_file_name, options):
    cfg = load_config(app_file_name, options)

    cfg.outdir = Path(cfg.dataset.path + '_' + Path(cfg.model.path).stem)
    cfg.outdir = Path(cfg.outdir).expanduser()

    if cfg.suffix not in ('.h5', '.tfrecord'):
        raise ValueError('Invalid suffix for output file, must either be h5 or tfrecord.')

    cfg.logdir = cfg.outdir
    cfg.logfile = cfg.outdir.joinpath('log.txt')
    cfg.outfile = cfg.outdir.joinpath('embeddings').with_suffix(cfg.suffix)

    # set seed for random number generators
    set_seed(cfg.seed)

    # write arguments and store some git revision info in a text files in the log directory
    ioutils.write_arguments(cfg, cfg.logdir.joinpath(Path(app_file_name).stem + '.yaml'))
    ioutils.store_revision_info(cfg.logdir)

    return cfg


class TrainClassifier(Config):
    def __init__(self, config):
        Config.__init__(self, config['config'])

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
        ioutils.store_revision_info(self.log_dir)


class Validate(Config):
    def __init__(self, options):
        Config.__init__(self, options['config'])

        if not self.seed:
            self.seed = 0
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        if not self.model.path:
            self.model.path = default_model

        self.model.normalize = True

        self.logs = self.model.path.joinpath('logs')
        self.txtfile = self.logs.joinpath('report.txt')

        if not self.batch_size:
            self.batch_size = default_batch_size

        if not self.dataset.path:
            self.dataset.path = default_test_dataset

        if not self.image.size:
            self.image.size = image_size

        if not self.image.normalization:
            self.image.normalization = image_normalization

        # write arguments and store some git revision info in a text files in the log directory
        ioutils.write_arguments(self, self.logs.joinpath('arguments.yaml'))
        ioutils.store_revision_info(self.logs)


