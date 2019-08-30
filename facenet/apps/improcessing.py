"""Performs processing images."""
# MIT License
# 
# Copyright (c) 2016 Ruslan N. Kosarev

import sys
import argparse
import numpy as np
import pathlib as plib
from PIL import Image

from facenet import config
from facenet import ioutils, h5utils, dataset

config = config.DefaultConfig()


def main(args):
    args.output_dir = plib.Path(args.output_dir).expanduser()
    ioutils.makedirs(args.output_dir)

    if args.h5file is None:
        args.h5file = str(args.output_dir) + '.h5'
    args.h5file = plib.Path(args.h5file).expanduser()

    dbase = dataset.DBase(args.input_dir)
    print(dbase)

    width = []
    height = []
    nrof_processed_images = 0

    for image_path in dbase.files:
        try:
            # this function returns PIL.Image object
            img = ioutils.read_image(image_path)
        except (IOError, ValueError, IndexError) as e:
            print('{}: {}'.format(image_path, e))
        else:

            output = img.resize(args.size, Image.ANTIALIAS)

            out_filename = args.output_dir.joinpath(image_path.parent.stem, image_path.stem + '.png')
            ioutils.write_image(output, out_filename)

            # write statistics
            height.append(img.size[0])
            width.append(img.size[1])

            h5utils.write(args.h5file, h5utils.filename2key(out_filename, 'size'), img.size)

    print('Total number of classes and images: {}/{}'.format(dbase.nrof_classes, dbase.nrof_images))
    print('Number of successfully processed images {}'.format(nrof_processed_images))

    report_file = args.report
    if report_file is None:
        input_dir = plib.Path(args.input_dir).expanduser()
        report_file = input_dir.parent.joinpath('{}_statistics.txt'.format(input_dir.name))

    print('Report txt file with statistics', report_file)

    ioutils.store_revision_info(report_file, ' '.join(sys.argv), mode='at')

    with open(report_file, 'at') as f:
        f.write('dataset {}\n'.format(plib.Path(args.input_dir).expanduser()))
        f.write('number of processed images {}\n'.format(nrof_processed_images))
        f.write('detector {}\n'.format(args.detector))
        f.write('\n')

        levels = np.linspace(0, 1, 11)

        for name, array in zip(['width', 'height'], [width, height]):
            f.write('statistics for {}\n'.format(name))
            f.write('mean {}\n'.format(np.mean(array)))
            f.write('median {}\n'.format(np.median(array)))
            f.write('quantiles {} /{}\n'.format(np.quantile(array, levels), levels))
            f.write('range {}\n'.format([min(array), max(array)]))
            f.write('\n')

    for x in [width, height]:
        print('mean ', np.mean(x))
        print('median ', np.median(x))
        print('range ', [np.min(x), np.max(x)])


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str,
                        help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str,
                        help='Directory to save processed images.')
    parser.add_argument('--report', type=str,
                        help='Output text file to save statistics.', default=None)
    parser.add_argument('--h5file', type=str,
                        help='Path to h5 file to write information about extracted faces.', default=None)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=config.image_size)
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    main(parse_arguments(sys.argv))
