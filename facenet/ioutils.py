# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import time
from PIL import Image
import numpy as np
import pathlib as plib


def makedirs(dirname):
    dirname = plib.Path(dirname)

    if not dirname.exists():
        dirname.mkdir(parents=True)


def write_image(image, filename, prefix=None, mode='RGB'):
    if prefix is not None:
        filename = os.path.join(str(prefix), str(filename))

    makedirs(os.path.dirname(filename))

    if isinstance(image, np.ndarray):
        image = array2pil(image, mode=mode)

    if image.save(str(filename)):
        raise IOError('while writing the file {}'.format(filename))


def read_image(filename, prefix=None):
    if prefix is not None:
        image = Image.open(os.path.join(str(prefix), str(filename)))
    else:
        image = Image.open(str(filename))

    return image


class ImageLoader:
    def __iter__(self):
        return self

    def __init__(self, input, prefix=None, display=100, log=True):

        if not isinstance(input, (str, list)):
            raise IOError('Input \'{}\' must be directory or list of files'.format(input))

        if isinstance(input, list):
            self.files = input
        elif os.path.isdir(os.path.expanduser(input)):
            prefix = os.path.expanduser(input)
            self.files = os.listdir(prefix)
        else:
            raise IOError('Directory \'{}\' does not exist'.format(input))

        self.counter = 0
        self.start_time = time.time()
        self.display = display
        self.size = len(self.files)
        self.prefix = str(prefix)
        self.log = log
        self.__filename = None

        print('Loader <{}> is initialized, number of files {}'.format(self.__class__.__name__, self.size))

    def __next__(self):
        if self.counter < self.size:
            if (self.counter + 1) % self.display == 0:
                elapsed_time = (time.time() - self.start_time) / self.display
                print('\rnumber of processed images {}/{}, {:.5f} seconds per image'.format(self.counter+1, self.size, elapsed_time), end='')
                self.start_time = time.time()

            image = read_image(self.files[self.counter], prefix=self.prefix)
            self.filename = image.filename

            if self.log:
                print('{}/{}, {}, {}'.format(self.counter, self.size, self.filename, image.size))

            self.counter += 1
            return image
        else:
            print('\n\rnumber of processed images {}'.format(self.size))
            raise StopIteration

    def reset(self):
        self.counter = 0
        return self


def pil2array(image, mode='RGB'):

    if image.mode == mode.upper():
        return np.array(image)

    output = []

    for channel in mode.upper():
        output.append(np.array(image.getchannel(channel)))

    output = np.stack(output, axis=2)

    return output


def array2pil(image, mode='RGB'):

    default_mode = 'RGB'
    index = []

    for sym in mode.upper():
        index.append(default_mode.index(sym))

    output = Image.fromarray(image[:, :, index], mode=default_mode)

    return output

