"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 Ruslan N. Kosarev

import sys
import os
import argparse

from facenet import ioutils
from facenet import facenet
from facenet.detectors.face_detector import image_processing, FaceDetector


def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)
    
    print('Creating networks and loading parameters')
    detector = FaceDetector(detector=args.detector, gpu_memory_fraction=args.gpu_memory_fraction)
    print(detector)

    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes.txt')
    
    with open(bounding_boxes_filename, 'w') as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0

        for cls in dataset:
            # define output class directory if exists skip this class
            output_class_dir = os.path.join(output_dir, cls.name)
            ioutils.makedirs(output_class_dir)

            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')

                print(image_path)

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
                        text_file.write('{}\n'.format(output_filename))
                    elif nrof_faces > 1 and not args.detect_multiple_faces:
                        print('Unable to align "{}"'.format(image_path))
                        text_file.write('{}\n'.format(output_filename))
                    else:
                        nrof_successfully_aligned += 1
                        filename_base, file_extension = os.path.splitext(output_filename)

                        for i, box in enumerate(bounding_boxes):
                            output = image_processing(img, box, args.image_size, margin=args.margin)

                            if i == 0:
                                output_filename_i = output_filename
                            else:
                                output_filename_i = '{}_{}{}'.format(filename_base, i, file_extension)

                            ioutils.write_image(output, output_filename_i)
                            text_file.write('{} {}\n'.format(output_filename_i, box.info()))

    print('Total number of images: {}'.format(nrof_images_total))
    print('Number of successfully aligned images: {}'.format(nrof_successfully_aligned))
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str,
                        help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str,
                        help='Directory with aligned face thumbnails.')
    parser.add_argument('--detector', type=str,
                        help='Detector to extract faces, pypimtcnn or frcnnv3.', default='frcnnv3')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)

    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    main(parse_arguments(sys.argv))
