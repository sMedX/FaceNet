
import os
import pathlib as plib
import numpy as np
from facenet import utils, h5utils


class ImageClass:
    """
    Stores the paths to images for a given class
    """
    def __init__(self, name, files, count=None):
        self.name = name
        self.count = count

        self.files = files
        self.files.sort()

    def __str__(self):
        return self.name + ', ' + str(len(self.files)) + ' images'

    def __len__(self):
        return len(self.files)


def list_names(dirname):
    names = os.listdir(dirname)
    names.sort()
    return names


def list_files(dirname, extension=None):
    dirname = os.path.expanduser(dirname)
    files = []

    if os.path.isdir(dirname):
        for file in list_names(dirname):
            if extension is None:
                files.append(os.path.join(dirname, file))
            else:
                _, ext = os.path.splitext(file)
                if ext == extension:
                    files.append(os.path.join(dirname, file))
    return files


def dataset(path, nrof_classes=0, has_class_directories=True, h5file=None):
    h5file = str(plib.Path(h5file).expanduser())

    ds = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in list_names(path_exp) if os.path.isdir(os.path.join(path_exp, path))]

    if nrof_classes > 0:
        classes = classes[:nrof_classes]

    for class_name in classes:
        dirname = os.path.join(path_exp, class_name)
        files = list_files(dirname, extension=config.file_extension)

        if h5file is not None:
            files = [f for f in files if h5utils.read(h5file, h5utils.filename2key(f, 'is_valid'), default=True)]

        if len(files) > 0:
            ds.append(ImageClass(class_name, files))

    return ds


class DBase:
    def __init__(self, path, extension='', h5file=None, nrof_classes=0):
        self.path = plib.Path(path).expanduser()

        self.h5file = h5file
        if self.h5file is not None:
            self.h5file = plib.Path(self.h5file).expanduser()

        classes = [path for path in self.path.glob('*') if path.is_dir()]
        classes.sort()

        if nrof_classes > 0:
            classes = classes[:nrof_classes]

        self.classes = []
        self.labels = []

        for count, class_path in enumerate(classes):
            files = class_path.glob('*' + extension)

            if self.h5file is not None:
                files = [f for f in files if h5utils.read(self.h5file, h5utils.filename2key(f, 'is_valid'), default=True)]
            else:
                files = [f for f in files]

            if len(files) > 0:
                self.classes.append(ImageClass(class_path.stem, files, count=count))

            print(classes[-1].name, end=utils.end(count, len(classes)))

            self.labels += [count]*len(files)

    def __repr__(self):
        """Representation of the database"""
        info = ('class {}\n'.format(self.__class__.__name__) +
                'Directory to load images {}\n'.format(self.path) +
                'h5 file to filter images {}\n'.format(self.h5file) +
                'Number of classes {} \n'.format(self.nrof_classes) +
                'Numbers of images {} and pairs {}\n'.format(self.nrof_images, self.nrof_pairs))
        return info

    @property
    def nrof_classes(self):
        return len(self.classes)

    @property
    def nrof_images(self):
        return sum(len(x) for x in self.classes)

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
    def files_as_strings(self):
        return [str(s) for s in self.files]

    def extract_data(self, folder_idx, embeddings=None):
        indices = np.where(self.labels == folder_idx)[0]
        files = [self.files[idx] for idx in indices]

        if embeddings is None:
            return files
        else:
            return files, embeddings[indices]
