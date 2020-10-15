# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import numpy as np
import h5py
from pathlib import Path


def write_dict(file, dct, group=None):
    with h5py.File(str(file), mode='a') as hf:
        def _write(dct, group=None):
            group = group + '/' if group else ''

            for key, item in dct.items():
                name = group + key
                if isinstance(item, dict):
                    _write(item, name)
                else:
                    data = np.atleast_1d(item)
                    if name in hf:
                        hf[name].resize(hf[name].shape[0] + data.shape[0], axis=0)
                        hf[name][-data.shape[0]:] = data
                    else:
                        hf.create_dataset(name, data=data, maxshape=(None,), compression='gzip', dtype=data.dtype)

        _write(dct, group=group)


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


def write(file, name, data, mode='a'):
    file = Path(file).expanduser()
    name = str(name)
    data = np.atleast_1d(data)

    with h5py.File(file, mode=mode) as hf:
        if name in hf:
            del hf[name]
        hf.create_dataset(name, data=data, compression='gzip', dtype=data.dtype)


def read(file, name, default=None):
    with h5py.File(str(file), mode='r') as hf:
        if name in hf:
            return hf[name][...]
        else:
            if default is not None:
                return default
            else:
                raise KeyError(f'Invalid key {name} in H5 file {file}')


def keys(file):
    with h5py.File(str(file), mode='r') as f:
        return list(f.keys())


def visit(file, func=print):
    with h5py.File(str(file), mode='r') as f:
        f.visit(func)


def visititems(file, func=None):
    items = []
    if func is None:
        def func(name, obj):
            if isinstance(obj, h5py.Dataset):
                items.append({'name': name, 'shape': obj.shape, 'type': obj.dtype})

    with h5py.File(str(file), mode='r') as f:
        f.visititems(func)

    return items
