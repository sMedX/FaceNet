
import os
import numpy as np
from scipy import spatial


class Database:
    def __init__(self, dirname, nrof_folders=0):
        self.dirname = os.path.expanduser(dirname)

        files, labels, dirs = get_files(self.dirname, nrof_folders=nrof_folders)

        self.files = files
        self.labels = np.array(labels)
        self.dirs = dirs
        self.nrof_folders = len(dirs)

    def __repr__(self):
        """Representation of the database"""
        info = ('class {}\n'.format(self.__class__.__name__) +
                'Directory to load images {}\n'.format(self.dirname) +
                'Numbers of folders {} \n'.format(self.nrof_folders) +
                'Numbers of images {} and pairs {}\n'.format(self.nrof_images, self.nrof_pairs))
        return info

    @property
    def nrof_images(self):
        return len(self.files)

    @property
    def nrof_pairs(self):
        return int(self.nrof_images * (self.nrof_images - 1) / 2)


def get_files(dirname, nrof_folders=0):

    if os.path.exists(dirname) is False:
        raise ValueError('Specified directory {} does not exist'.format(dirname))

    list_of_files = []
    list_of_dirs = []

    list_of_labels = []
    count = 0

    if nrof_folders == 0:
        nrof_folders = np.Inf

    for root, dirs, files in os.walk(dirname):
        if len(dirs) < nrof_folders:
            if len(dirs) == 0 and len(list_of_dirs) < nrof_folders:
                list_of_files += [os.path.join(root, file) for file in files]
                list_of_dirs.append(root)

                list_of_labels += [count]*len(files)
                count += 1

    return list_of_files, list_of_labels, list_of_dirs


def label_array(labels):

    if isinstance(labels, (np.ndarray, list)) is False:
        raise ValueError('label_array: input labels must be list or numpy.ndarray')

    if isinstance(labels, list):
        labels = np.array([labels]).transpose()

    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=0).transpose()

    labels = spatial.distance.pdist(labels, metric='sqeuclidean')
    labels = np.array(labels < 0.5, np.int)

    return labels


def label_matrix_(image_paths, diagonal=True):

    basenames = [os.path.basename(os.path.dirname(path)) for path in image_paths]

    labels = np.zeros([len(image_paths), len(image_paths)], dtype=np.uint8)

    for i, basename1 in enumerate(basenames):
        for k, basename2 in enumerate(basenames[:i]):
            if basename1 == basename2:
                labels[i][k] = labels[k][i] = 1

        if diagonal:
            labels[i][i] = 1

    return labels

