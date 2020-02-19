from pathlib import Path
import numpy as np
import math
from facenet import utils, h5utils


class ImageClass:
    """
    Stores the paths to images for a given class
    """

    def __init__(self, config, files=None, ext=''):
        self.path = Path(config.path).expanduser()
        self.name = self.path.stem

        if files is None:
            files = [str(f) for f in self.path.glob('*' + ext)]

            if config.h5file is not None:
                h5file = Path(config.h5file).expanduser()
                files = [f for f in files if h5utils.read(h5file, h5utils.filename2key(f, 'is_valid'), default=True)]

            if config.nrof_images is not None:
                if len(files) > config.nrof_images:
                    files = np.random.choice(files, size=config.nrof_images, replace=False)

        self.files = files
        self.files.sort()

    def __repr__(self):
        return 'name: {}, images: {}'.format(self.name, self.nrof_images)

    @property
    def nrof_images(self):
        return len(self.files)

    @property
    def nrof_pairs(self):
        return self.nrof_images * (self.nrof_images - 1) // 2

    def random_choice(self, nrof_images_per_class):
        files = self.files
        if self.nrof_images > nrof_images_per_class:
            files = np.random.choice(files, size=nrof_images_per_class, replace=False)
        return ImageClass(self, files=files)


class DBase:
    def __init__(self, config, classes=None, ext=''):
        self.path = Path(config.path).expanduser()
        self.h5file = config.h5file
        if self.h5file is not None:
            self.h5file = Path(self.h5file).expanduser()

        if classes is None:
            dirs = [p for p in self.path.glob('*') if p.is_dir()]
            if config.nrof_classes is not None:
                dirs = np.random.choice(dirs, size=config.nrof_classes, replace=False)
            dirs.sort()

            classes = []

            for idx, path in enumerate(dirs):
                config.path = path
                image_class = ImageClass(config, ext=ext)
                if image_class.nrof_images > 0:
                    classes.append(ImageClass(config, ext=ext))
                    print('\r({}/{}) {}'.format(idx, len(dirs), classes[-1].__repr__()), end=utils.end(idx, len(dirs)))

        self.classes = classes

    @property
    def labels(self):
        labels = []
        for idx, cls in enumerate(self.classes):
            labels += [idx] * cls.nrof_images
        return np.array(labels)

    def __repr__(self):
        """Representation of the database"""
        info = 'class {}\n'.format(self.__class__.__name__) + \
               'Directory to load images {}\n'.format(self.path) + \
               'h5 file to filter images {}\n'.format(self.h5file) + \
               'Number of classes {} \n'.format(self.nrof_classes) + \
               'Number of images {}\n'.format(self.nrof_images) + \
               'Number of pairs {}\n'.format(self.nrof_pairs) + \
               'Number of positive pairs {} ({:.6f} %)\n'.format(self.nrof_positive_pairs, 100 * self.nrof_positive_pairs / self.nrof_pairs) + \
               'Number of negative pairs {} ({:.6f} %)\n'.format(self.nrof_negative_pairs, 100 * self.nrof_negative_pairs / self.nrof_pairs) + \
               'Minimal number of images in class {}\n'.format(self.min_nrof_images) + \
               'Maximal number of images in class {}\n'.format(self.max_nrof_images)

        return info

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
    def files(self):
        f = []
        for cls in self.classes:
            f += cls.files
        return f

    @property
    def files_as_posix(self):
        f = []
        for cls in self.classes:
            f += cls.files_as_posix
        return f

    def random_choice(self, nrof_images_per_class, nrof_classes=None):

        class_indices = np.arange(self.nrof_classes)
        if nrof_classes is not None:
            if self.nrof_classes > nrof_classes:
                class_indices = np.random.choice(class_indices, size=nrof_classes, replace=False)
                class_indices.sort()

        classes = []
        for i in class_indices:
            classes.append(self.classes[i].random_choice(nrof_images_per_class))

        return DBase(self, classes=classes)

    def extract_data(self, folder_idx, embeddings=None):
        indices = np.where(self.labels == folder_idx)[0]
        files = [self.files[idx] for idx in indices]

        if embeddings is None:
            return files
        else:
            return files, embeddings[indices]

    def split(self, split_ratio, min_nrof_images_per_class, mode='images'):
        if split_ratio <= 0.0:
            return self.classes, []

        if mode == 'classes':
            nrof_classes = len(self.classes)
            class_indices = np.arange(nrof_classes)
            np.random.shuffle(class_indices)
            split = int(round(nrof_classes * (1 - split_ratio)))
            train_set = [self.classes[i] for i in class_indices[0:split]]
            test_set = [self.classes[i] for i in class_indices[split:-1]]
        elif mode == 'images':
            train_set = []
            test_set = []
            for cls in self.classes:
                paths = cls.files
                np.random.shuffle(paths)
                nrof_images_in_class = len(paths)
                split = int(math.floor(nrof_images_in_class * (1 - split_ratio)))
                if split == nrof_images_in_class:
                    split = nrof_images_in_class - 1
                if split >= min_nrof_images_per_class and nrof_images_in_class - split >= 1:
                    train_set.append(ImageClass(cls.name, paths[:split]))
                    test_set.append(ImageClass(cls.name, paths[split:]))
        else:
            raise ValueError('Invalid train/test split mode "%s"' % mode)

        return train_set, test_set


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].files
        labels_flat += [i] * len(dataset[i].files)
    return image_paths_flat, labels_flat