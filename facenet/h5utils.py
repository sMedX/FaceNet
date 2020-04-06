# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np
import h5py
from pathlib import Path


def write_dict(file, dct, group=None):
    file = Path(file).expanduser()
    print('save statistics to the h5 file {}'.format(file))

    with h5py.File(str(file), 'a') as f:
        for name, data in dct.items():
            if group:
                name = group + '/' + name

            data = np.atleast_1d(data)

            if name in f:
                f[name][...] = data
            else:
                f.create_dataset(name, data=data, compression='gzip', dtype=data.dtype)


def filename2key(filename, key):
    file = Path(filename)
    return str(Path(file.parent.stem).joinpath(file.stem, key))


def write_image(hf, name, image, mode='a', check_name=True):
    with h5py.File(str(hf), mode) as hf:

        if name in hf and check_name is True:
            raise IOError('data set {} has already existed'.format(name))

        if not name in hf:
            dset = hf.create_dataset(name=name,
                                     data=image,
                                     dtype='uint8',
                                     compression='gzip',
                                     compression_opts=9)
        else:
            hf[name][...] = image


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
