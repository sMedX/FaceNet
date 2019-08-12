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
        args.outdir = args.dir + '_false_examples'
    ioutils.makedirs(args.outdir)

    if args.h5file is None:
        args.h5file = args.dir + '.h5'
    args.h5file = plib.Path(args.h5file).expanduser()

    # Get the paths for the corresponding images
    tf_files = dataset.list_files(args.dir, extension='.tfrecord')
    print('dataset', args.dir)
    print('number of tf records', len(tf_files))

    for i, file1 in enumerate(tf_files):
        print('\r{}/{}'.format(i, len(tf_files)), end=utils.end(i, len(tf_files)))

        tf1 = utils.TFRecord(file1)
        for file2 in tf_files[i+1:]:
            tf2 = utils.TFRecord(file2)
            dist = 2*(1 - tf1.embeddings @ tf2.embeddings.transpose())

            while np.nanmin(dist) < args.threshold:
                n, m = np.unravel_index(np.nanargmin(dist), dist.shape)
                identical_faces = (tf1.files[n], tf2.files[m])

                image = utils.ConcatenateImages(identical_faces[0], identical_faces[1], dist[n, m])
                image.save(args.outdir)

                print()
                print(dist[n, m])
                print(identical_faces[0])
                print(identical_faces[1])

                dist[n, m] = np.nan

                if args.h5file:
                    for file in identical_faces:
                        file = plib.Path(file)
                        key = plib.Path(file.parent.stem).joinpath(file.stem, 'is_valid')
                        h5utils.write(args.h5file, key, False)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dir', type=str,
        help='Path to the directory with tf records files.')
    parser.add_argument('--outdir', type=str,
        help='Directory to save examples with identical images.', default=None)
    parser.add_argument('--h5file', type=str,
        help='Path to h5 file to write information about false images.', default=None)
    parser.add_argument('--threshold', type=float,
        help='Threshold to classify identical faces.', default=0.10)
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    main(parse_arguments(sys.argv))
