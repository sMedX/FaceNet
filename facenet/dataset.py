# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

from tqdm import tqdm
from pathlib import Path
from loguru import logger

import tensorflow as tf
import numpy as np

from facenet import h5utils


class DefaultConfig:
    def __init__(self, path, h5file=None, nrof_classes=None, min_nrof_images=1, max_nrof_images=None):
        self.path = path
        # Path to h5 file with information about valid images.
        self.h5file = h5file
        # Number of classes to download from data set
        self.nrof_classes = nrof_classes
        # Minimal number of images to download from class
        self.min_nrof_images = min_nrof_images
        # Maximal number of images to download from class
        self.max_nrof_images = max_nrof_images


def tf_dataset_api(files, labels, loader, batch_size, shuffle=False, buffer_size=None, repeat=False):

    if shuffle:
        data = list(zip(files, labels))
        np.random.shuffle(data)
        files, labels = map(list, zip(*data))

    images = tf.data.Dataset.from_tensor_slices(files).map(loader, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip((images, labels))

    if buffer_size is not None:
        ds = ds.shuffle(buffer_size=buffer_size*batch_size, reshuffle_each_iteration=True)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    info = (f'{ds}\n' +
            f'batch size: {batch_size}\n' +
            f'shuffle: {shuffle}\n' +
            f'buffer size: {buffer_size}\n' +
            f'repeat: {repeat}\n' +
            f'cardinality: {ds.cardinality()}')

    logger.info('\n' + info)

    return ds


class ImageClass:
    """
    Stores the paths to images for a given class
    """

    def __init__(self, config, files=None, ext=''):
        self.config = config
        self.path = Path(config.path).expanduser()
        self.name = self.path.stem

        if files is None:
            files = list(self.path.glob('*' + ext))

            if config.h5file:
                h5file = Path(config.h5file).expanduser()
                files = [f for f in files if h5utils.read(h5file, h5utils.filename2key(f, 'is_valid'), default=True)]

            if config.max_nrof_images:
                if len(files) > config.max_nrof_images:
                    files = np.random.choice(files, size=config.max_nrof_images, replace=False)

        self.files = [str(f) for f in files]
        self.files.sort()

    def __repr__(self):
        return '{} ({}/{})'.format(self.__class__.__name__, self.name, self.nrof_images)

    def __bool__(self):
        return True if self.nrof_images > self.config.min_nrof_images else False

    @property
    def nrof_images(self):
        return len(self.files)

    @property
    def nrof_pairs(self):
        return self.nrof_images * (self.nrof_images - 1) // 2

    def random_choice(self, nrof_images):
        files = self.files
        if nrof_images < self.nrof_images:
            files = np.random.choice(files, size=nrof_images, replace=False)
        return ImageClass(self, files=files)

    def random_split(self, split_ratio=0.05, nrof_images=None):
        index = round(self.nrof_images * split_ratio)
        if nrof_images:
            index = min(index, nrof_images)

        files = np.random.permutation(self.files)

        return ImageClass(self, files=files[index:]), ImageClass(self, files=files[:index])


class DBase:
    def __init__(self, config, classes=None, ext=''):
        if isinstance(config, Path):
            config = DefaultConfig(config)

        if not config.min_nrof_images:
            config.min_nrof_images = 1

        self.path = config.path
        self.h5file = config.h5file

        if classes is None:
            if not self.path:
                raise ValueError('Path to download dataset does not specified.')

            self.path = Path(self.path).expanduser()
            print('Download data set from {}'.format(self.path))

            if not self.path.exists():
                raise ValueError(f'Directory {self.path} does not exist')

            if self.h5file:
                self.h5file = Path(self.h5file).expanduser()

            dirs = [p for p in self.path.glob('*') if p.is_dir()]
            if config.nrof_classes:
                if len(dirs) > config.nrof_classes:
                    dirs = np.random.choice(dirs, size=config.nrof_classes, replace=False)
            dirs.sort()

            classes = []

            with tqdm(total=len(dirs)) as bar:
                for idx, path in enumerate(dirs):
                    config.path = path
                    images = ImageClass(config, ext=ext)
                    if images:
                        classes.append(images)
                    bar.set_postfix_str('{}'.format(str(images)))
                    bar.update()

        self.classes = classes
        logger.info(self)

    def __repr__(self):
        """Representation of the database"""
        info = (f'{self.__class__.__name__}\n' +
                f'{self.path}\n' +
                f'h5 file {self.h5file}\n' +
                f'Number of classes {self.nrof_classes} \n' +
                f'Number of images {self.nrof_images}\n' +
                f'Number of pairs {self.nrof_pairs}\n' +
                f'Number of positive pairs {self.nrof_positive_pairs} \n' +
                f'Number of negative pairs {self.nrof_negative_pairs} \n' +
                f'Minimal number of images in class {self.min_nrof_images}\n' +
                f'Maximal number of images in class {self.max_nrof_images}\n')
        return info

    def __bool__(self):
        return True if self.nrof_classes > 1 else False

    @property
    def files(self):
        files = []
        for cls in self.classes:
            files += cls.files
        return files

    @property
    def labels(self):
        labels = []
        for idx, cls in enumerate(self.classes):
            labels += [idx] * cls.nrof_images
        return np.array(labels)

    @property
    def min_nrof_images(self):
        return min(cls.nrof_images for cls in self.classes)

    @property
    def max_nrof_images(self):
        return max(cls.nrof_images for cls in self.classes)

    @property
    def nrof_classes(self):
        return len(self.classes)

    @property
    def nrof_images(self):
        return sum(cls.nrof_images for cls in self.classes)

    @property
    def nrof_negative_pairs(self):
        return self.nrof_pairs - self.nrof_positive_pairs

    @property
    def nrof_positive_pairs(self):
        return sum(cls.nrof_pairs for cls in self.classes)

    @property
    def nrof_pairs(self):
        return self.nrof_images * (self.nrof_images - 1) // 2

    @property
    def nrof_images_per_class(self):
        return [cls.nrof_images for cls in self.classes]

    def random_choice(self, split_ratio, nrof_classes=None):

        class_indices = np.arange(self.nrof_classes)
        if nrof_classes is not None:
            if self.nrof_classes > nrof_classes:
                class_indices = np.random.choice(class_indices, size=nrof_classes, replace=False)
                class_indices.sort()

        classes = []
        for i in class_indices:
            nrof_images = round(self.classes[i].nrof_images * split_ratio)
            if nrof_images > 0:
                classes.append(self.classes[i].random_choice(nrof_images))

        return DBase(self, classes=classes)

    def random_split(self, config):
        train = []
        test = []

        for cls in self.classes:
            train_, test_ = cls.random_split(config.split_ratio, nrof_images=config.nrof_images)
            train.append(train_)
            test.append(test_)

        return DBase(self, classes=train), DBase(self, classes=test)

    # def extract_data(self, folder_idx, embeddings=None):
    #     indices = np.where(self.labels == folder_idx)[0]
    #     files = [self.files[idx] for idx in indices]
    #
    #     if embeddings is None:
    #         return files
    #     else:
    #         return files, embeddings[indices]
