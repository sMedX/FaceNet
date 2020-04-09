# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np
import h5py
from pathlib import Path


def write_dict(file, dct, group=None):
    group = group + '/' if group else ''

    with h5py.File(str(file), mode='a') as hf:
        for key, data in dct.items():
            name = group + key
            data = np.atleast_1d(data)

            if name in hf:
                hf[name].resize(hf[name].shape[0] + data.shape[0], axis=0)
                hf[name][-data.shape[0]:] = data
            else:
                hf.create_dataset(name, data=data, maxshape=(None,), compression='gzip', dtype=data.dtype)


def filename2key(filename, key):
    file = Path(filename)
    return str(Path(file.parent.stem).joinpath(file.stem, key))


def write_image(hf, name, image, mode='a', check_name=True):
    with h5py.File(str(hf), mode) as hf:

        if name in hf and check_name:
            raise IOError('data set {} has already existed'.format(name))

        if name in hf:
            hf[name][...] = image
        else:
            hf.create_dataset(name=name, data=image, dtype='uint8', compression='gzip', compression_opts=9)


def write(filename, name, data, mode='a'):
    filename = os.path.expanduser(str(filename))
    name = str(name)

    with h5py.File(filename, mode=mode) as hf:
        if name in hf:
            del hf[name]

        data = np.atleast_1d(data)

        hf.create_dataset(name,
                          data=data,
                          compression='gzip',
                          dtype=data.dtype)

    # print('dataset \'{}\' has been written to the file {} (length {})'.format(name, filename, len(data)))


def read(file, name, default=None):
    with h5py.File(str(file), mode='r') as hf:
        if name in hf:
            return hf[name][...]
        else:
            return default


def keys(file):
    with h5py.File(str(file), mode='r') as f:
        return list(f.keys())


def visit(file, func=print):
    with h5py.File(str(file), mode='r') as f:
        f.visit(func)
