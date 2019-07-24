# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import time
import numpy as np
import pathlib as plib
import datetime
import tensorflow as tf
from PIL import Image
from subprocess import Popen, PIPE


def store_revision_info(src_path, output_filename, arg_string, mode='w'):
    if os.path.isdir(output_filename):
        output_filename = os.path.join(output_filename, 'revision_info.txt')

    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' + e.strerror

    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' + e.strerror

    # Store a text file in the log directory
    with open(output_filename, mode) as f:
        f.write('{}\n'.format(datetime.datetime.now()))
        f.write('arguments: %s\n--------------------\n' % arg_string)
        f.write('tensorflow version: %s\n--------------------\n' % tf.__version__)  # @UndefinedVariable
        f.write('git hash: %s\n--------------------\n' % git_hash)
        f.write('git diff: %s\n' % git_diff)
        f.write('\n')


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


def read_image(filename, prefix=None, mode=None):
    if prefix is not None:
        image = Image.open(os.path.join(str(prefix), str(filename)))
    else:
        image = Image.open(str(filename))

    if mode is not None:
        image = image.convert(mode)

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
    return np.array(image.convert(mode.upper()))


def array2pil(image, mode='RGB'):

    default_mode = 'RGB'
    index = []

    for sym in mode.upper():
        index.append(default_mode.index(sym))

    output = Image.fromarray(image[:, :, index], mode=default_mode)

    return output

