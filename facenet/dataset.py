
import os
import numpy as np


class Dataset:
    def __init__(self, dirname, nrof_folders=0):
        self.dirname = os.path.expanduser(dirname)

        self.files = []
        self.dirs = []
        self.labels = []

        self._get_files(self.dirname, nrof_folders=nrof_folders)

        self.nrof_folders = len(self.dirs)

    def __repr__(self):
        """Representation of the database"""
        info = ('class {}\n'.format(self.__class__.__name__) +
                'Directory to load images {}\n'.format(self.dirname) +
                'Number of folders {} \n'.format(self.nrof_folders) +
                'Numbers of images {} and pairs {}\n'.format(self.nrof_images, self.nrof_pairs))
        return info

    @property
    def name(self):
        return os.path.basename(self.dirname)

    def filenames(self, mode='with_dir'):
        if mode == 'with_dir':
            return [os.path.join(os.path.basename(os.path.dirname(file)), os.path.basename(file)) for file in self.files]
        else:
            raise 'Undefined mode {} to return list of file names'.format(mode)

    @property
    def nrof_images(self):
        return len(self.files)

    @property
    def nrof_pairs(self):
        return int(self.nrof_images * (self.nrof_images - 1) / 2)

    def _get_files(self,dirname, nrof_folders=0):

        if os.path.exists(dirname) is False:
            raise ValueError('Specified directory {} does not exist'.format(dirname))

        count = 0

        if nrof_folders == 0:
            nrof_folders = np.Inf

        for root, dirs, files in os.walk(dirname):
            if len(dirs) < nrof_folders:
                if len(dirs) == 0 and len(files) != 0 and len(self.dirs) < nrof_folders:
                    self.files += [os.path.join(root, file) for file in files]
                    self.dirs.append(root)

                    self.labels += [count]*len(files)
                    count += 1

        self.labels = np.array(self.labels)

    def extract_data(self, folder_idx, embeddings=None):
        indices = np.where(self.labels == folder_idx)[0]
        files = [self.files[idx] for idx in indices]

        if embeddings is None:
            return files
        else:
            return files, embeddings[indices]
