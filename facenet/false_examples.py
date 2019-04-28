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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
from facenet import dataset, utils
from facenet.statistics import FalseExamples


def main(args):

    # Get the paths for the corresponding images
    dbase = dataset.Dataset(args.dir)
    print(dbase)

    tfrecord = utils.TFRecord(args.tfrecord)
    print(tfrecord)

    examples = FalseExamples(dbase, tfrecord,
                             threshold=args.threshold,
                             metric=args.distance_metric,
                             subtract_mean=args.subtract_mean)
    examples.write_false_pairs(args.false_positive_dir, args.false_negative_dir)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dir', type=str,
        help='Path to the data directory containing aligned face patches.')
    parser.add_argument('tfrecord', type=str,
        help='Path to tfrecord file with embeddings.')
    parser.add_argument('--threshold', type=float,
        help='Threshold value to identify same faces.', default=1)
    parser.add_argument('--distance_metric', type=int,
        help='Distance metric  0:euclidian, 1:cosine similarity.', default=0)
    parser.add_argument('--subtract_mean',
        help='Subtract feature mean before calculating distance.', action='store_true')
    parser.add_argument('--false_positive_dir', type=str,
        help='Directory to save false positive pairs.', default='')
    parser.add_argument('--false_negative_dir', type=str,
        help='Directory to save false negative pairs.', default='')
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    main(parse_arguments(sys.argv))
