# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

from cached_property import cached_property
import numpy as np
from pathlib import Path

from facenet import utils, h5utils


class ImageClass:
    """
    Stores the paths to images for a given class
    """

    def __init__(self, config, files=None, ext=''):
        self.path = Path(config.path).expanduser()
        self.name = self.path.stem

        if files is None:
            files = list(self.path.glob('*' + ext))

            if config.h5file:
                h5file = Path(config.h5file).expanduser()
                files = [f for f in files if h5utils.read(h5file, h5utils.filename2key(f, 'is_valid'), default=True)]

            if config.nrof_images is not None:
                if len(files) > config.nrof_images:
                    files = np.random.choice(files, size=config.nrof_images, replace=False)

        self.files = [str(f) for f in files]
        self.files.sort()

    def __repr__(self):
        return 'name: {}, images: {}'.format(self.name, self.nrof_images)

    def __bool__(self):
        return True if self.nrof_images > 1 else False

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

    def random_split(self, split_ratio=0.05):
        index = round(self.nrof_images * (1-split_ratio))
        return ImageClass(self, files=self.files[:index]), ImageClass(self, files=self.files[index:])


class DBase:
    def __init__(self, config, classes=None, ext=''):
        self.path = config.path
        self.h5file = config.h5file

        if classes is None:
            self.path = Path(self.path).expanduser()
            print('Download data set from {}'.format(self.path))

            if not self.path.exists():
                raise ValueError('Directory {} does not exit'.format(self.path))

            if self.h5file:
                self.h5file = Path(self.h5file).expanduser()

            dirs = [p for p in self.path.glob('*') if p.is_dir()]
            if config.nrof_classes:
                dirs = np.random.choice(dirs, size=config.nrof_classes, replace=False)
            dirs.sort()

            classes = []

            for idx, path in enumerate(dirs):
                config.path = path
                images = ImageClass(config, ext=ext)
                if images:
                    classes.append(images)
                    print('\r({}/{}) {}'.format(idx, len(dirs), str(images)), end=utils.end(idx, len(dirs)))

        self.classes = classes

    def __repr__(self):
        """Representation of the database"""
        return ('{}({})\n'.format(self.__class__.__name__, self.path) +
                'h5 file to filter images {}\n'.format(self.h5file) +
                'Number of classes {} \n'.format(self.nrof_classes) +
                'Number of images {}\n'.format(self.nrof_images) +
                'Number of pairs {}\n'.format(self.nrof_pairs) +
                'Number of positive pairs {} \n'.format(self.nrof_positive_pairs) +
                'Number of negative pairs {} \n'.format(self.nrof_negative_pairs) +
                'Minimal number of images in class {}\n'.format(self.min_nrof_images) +
                'Maximal number of images in class {}\n'.format(self.max_nrof_images))

    def __bool__(self):
        return True if self.nrof_classes > 1 else False

    @cached_property
    def files(self):
        files = []
        for cls in self.classes:
            files += cls.files
        return files

    @cached_property
    def labels(self):
        labels = []
        for idx, cls in enumerate(self.classes):
            labels += [idx] * cls.nrof_images
        return np.array(labels)

    @property
    def min_nrof_images(self):
        return min(cls.nrof_images for cls in self.classes) if self.nrof_classes > 0 else 0

    @property
    def max_nrof_images(self):
        return max(cls.nrof_images for cls in self.classes) if self.nrof_classes > 0 else 0

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

    def random_split(self, split_ratio=0.05):
        train = []
        test = []

        for cls in self.classes:
            train_, test_ = cls.random_split(split_ratio=split_ratio)
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
