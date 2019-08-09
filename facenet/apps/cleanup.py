"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2019 SMedX
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import argparse
import numpy as np
import pathlib as plib
from facenet import dataset, utils, ioutils, h5utils


def main(args):

    if args.outdir is None:
        args.outdir = args.dir + '_same_images'
    ioutils.makedirs(args.outdir)

    if args.h5file is None:
        args.h5file = args.dir + '.h5'
    args.h5file = plib.Path(args.h5file).expanduser()

    # Get the paths for the corresponding images
    files = dataset.list_files(args.dir, extension='.tfrecord')
    print('dataset', args.dir)
    print('number of tf records', len(files))

    for i, file1 in enumerate(files):
        tf1 = utils.TFRecord(file1)
        for k, file2 in enumerate(files[:i]):
            print('\r{}/{} ({})'.format(k, i, len(files)), end=utils.end(i, len(files)))
            tf2 = utils.TFRecord(file2)
            dist = tf1.embeddings @ tf2.embeddings.transpose()

            while dist.max() > 0.98:
                n, m = np.unravel_index(dist.argmax(), dist.shape)
                same_images = (tf1.files[n], tf2.files[m])

                image = utils.ConcatenateImages(same_images[0], same_images[1], dist[n, m])
                image.save(args.outdir)

                print()
                print(dist[n, m])
                print(tf1.files[n])
                print(tf2.files[m])

                dist[n, m] = -np.Inf

                for file in same_images:
                    file = plib.Path(file)
                    key = plib.Path(file.parent.stem).joinpath(file.stem, 'is_valid')
                    h5utils.write(args.h5file, key, False)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dir', type=str,
        help='Path to the data directory containing extracted face patches.')
    parser.add_argument('--outdir', type=str,
        help='Directory to save examples.', default=None)
    parser.add_argument('--h5file', type=str,
        help='Path to h5 file to save information about false images.', default=None)
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    main(parse_arguments(sys.argv))
