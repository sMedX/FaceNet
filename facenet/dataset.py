
import os
import numpy as np
from scipy import spatial


class Dataset:
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
                'Number of folders {} \n'.format(self.nrof_folders) +
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
