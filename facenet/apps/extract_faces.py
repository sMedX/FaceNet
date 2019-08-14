"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 Ruslan N. Kosarev

import sys
import argparse
import pathlib as plib

from facenet import dataset, ioutils, h5utils
from facenet.detectors.face_detector import image_processing, FaceDetector
from facenet import facenet


def main(args):
    args.output_dir = plib.Path(args.output_dir).expanduser()
    ioutils.makedirs(args.output_dir)

    if args.h5file is None:
        args.h5file = str(args.output_dir) + '.h5'
    args.h5file = plib.Path(args.h5file).expanduser()

    # store some git revision info in a text file in the log directory
    # src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(plib.Path(__file__).parent, args.output_dir, ' '.join(sys.argv))
    # dataset = facenet.get_dataset(args.input_dir)
    dbase = dataset.DBase(args.input_dir)
    
    print('Creating networks and loading parameters')
    detector = FaceDetector(detector=args.detector, gpu_memory_fraction=args.gpu_memory_fraction)
    print(detector)

    # bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes.txt')
    
    # with open(bounding_boxes_filename, 'w') as text_file:
    nrof_images_total = 0
    nrof_successfully_aligned = 0

    for cls in dbase.classes:
        # define output class directory if exists skip this class
        output_class_dir = args.output_dir.joinpath(cls.name)
        ioutils.makedirs(output_class_dir)

        for image_path in cls.files:
            print(image_path)

            nrof_images_total += 1
            out_filename = output_class_dir.joinpath(image_path.stem + '.png')

            try:
                # this function returns PIL.Image object
                img = ioutils.read_image(image_path, mode='RGB')
            except (IOError, ValueError, IndexError) as e:
                print(e)
            else:
                bounding_boxes = detector.detect(img)
                nrof_faces = len(bounding_boxes)

                if nrof_faces == 0:
                    print('Unable to find dace "{}"'.format(image_path))
                elif nrof_faces > 1 and not args.detect_multiple_faces:
                    print('The number of detected faces more than one "{}"'.format(image_path))
                else:
                    nrof_successfully_aligned += 1

                    for i, box in enumerate(bounding_boxes):
                        output = image_processing(img, box, args.image_size, margin=args.margin)

                        if i == 0:
                            out_filename_i = out_filename
                        else:
                            out_filename_i = out_filename.parent.joinpath('{}_{}{}'.format(out_filename.stem, i, out_filename.suffix))

                        ioutils.write_image(output, out_filename_i)
                        h5utils.write(args.h5file, h5utils.filename2key(out_filename_i, 'size'), (box.height, box.width))

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
    parser.add_argument('--h5file', type=str,
                        help='Path to h5 file to write information about extracted faces.', default=None)

    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    main(parse_arguments(sys.argv))
