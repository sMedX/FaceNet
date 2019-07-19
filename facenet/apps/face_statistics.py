"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 Ruslan N. Kosarev

import sys
import os
import argparse
import numpy as np
import random
import pathlib as plib

from facenet import ioutils
from facenet import facenet
from facenet.detectors.face_detector import FaceDetector


def main(args):
    # store some git revision info in a text file in the log directory
    dataset = facenet.get_dataset(args.input_dir)
    image_paths = []
    for cls in dataset:
        image_paths += cls.image_paths

    print('Creating networks and loading parameters')
    detector = FaceDetector(detector=args.detector, gpu_memory_fraction=args.gpu_memory_fraction)
    print(detector)

    width = []
    height = []
    
    nrof_processed_images = 0

    # if args.random_order
    # random.shuffle(dataset)

    if args.nrof_images:
        random.shuffle(image_paths)

    for image_path in image_paths:
        if nrof_processed_images == args.nrof_images:
            break

        try:
            # this function returns PIL.Image object
            img = ioutils.read_image(image_path)
        except (IOError, ValueError, IndexError) as e:
            print('{}: {}'.format(image_path, e))
        else:
            bounding_boxes = detector.detect(img)
            nrof_faces = len(bounding_boxes)

            if nrof_faces == 0:
                print('Unable to align "{}"'.format(image_path))
            elif nrof_faces > 1 and not args.detect_multiple_faces:
                print('Unable to align "{}"'.format(image_path))
            else:
                nrof_processed_images += 1

                for i, box in enumerate(bounding_boxes):
                    print(image_path, box.width, box.height)
                    width.append(box.width)
                    height.append(box.height)

    print('Total number of images: {}'.format(len(image_paths)))
    print('Number of successfully processed images {}'.format(nrof_processed_images))

    output_file = args.output
    if output_file is None:
        input_dir = plib.Path(args.input_dir).expanduser()
        output_file = input_dir.parent.joinpath('{}_{}_statistics.txt'.format(input_dir.name, args.detector))

    print('Output statistic file', output_file)

    src_path, _ = os.path.split(os.path.realpath(__file__))
    ioutils.store_revision_info(src_path, output_file, ' '.join(sys.argv), mode='at')

    with open(output_file, 'at') as f:
        f.write('dataset {}\n'.format(os.path.expanduser(args.input_dir)))
        f.write('number of processed images {}\n'.format(nrof_processed_images))
        f.write('detector {}\n'.format(args.detector))
        f.write('\n')

        for name, array in zip(['width', 'height'], [width, height]):
            f.write('statistics for {}\n'.format(name))
            f.write('mean {}\n'.format(np.mean(array)))
            f.write('median {}\n'.format(np.median(array)))
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
    parser.add_argument('--output', type=str,
                        help='Output text file to save face statistics.', default=None)
    parser.add_argument('--detector', type=str,
                        help='Detector to extract faces, pypimtcnn or frcnnv3.', default='frcnnv3')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    parser.add_argument('--nrof_images', type=int,
                        help='Number of random shuffled images to evaluate statistics.', default=10000)

    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    main(parse_arguments(sys.argv))
