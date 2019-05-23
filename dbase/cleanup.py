
import sys
import argparse


def main(args):
    pass


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
    parser.add_argument('--nrof_fpos_images', type=int,
                        help='Number of false positive pairs per folder.', default=10)
    parser.add_argument('--false_negative_dir', type=str,
                        help='Directory to save false negative pairs.', default='')
    parser.add_argument('--nrof_fneg_images', type=int,
                        help='Number of false negative pairs per folder.', default=2)
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    main(parse_arguments(sys.argv))
