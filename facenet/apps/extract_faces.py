"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# Copyright (c) 2020 Ruslan N. Kosarev

import sys
import click
from pathlib import Path
import numpy as np

from facenet import dataset, ioutils, h5utils
from facenet.detectors.face_detector import image_processing, FaceDetector
from facenet import facenet, config


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options of the application.')
def main(**args_):
    args = config.YAMLConfig(args_['config'])

    if args.outdir is None:
        args.outdir = '{}_{}extracted_{}'.format(args.dataset.path, args.detector, args.image_size)
    args.output_dir = Path(args.outdir).expanduser()
    ioutils.makedirs(args.output_dir)

    if args.h5file is None:
        args.h5file = args.output_dir.joinpath('statistics.h5')
    args.h5file = Path(args.h5file).expanduser()

    # store some git revision info in a text file in the log directory
    facenet.store_revision_info(Path(__file__).parent, args.output_dir, ' '.join(sys.argv))

    # write arguments and store some git revision info in a text files in the log directory
    ioutils.write_arguments(args, args.output_dir.joinpath('arguments.yaml'))
    ioutils.store_revision_info(args.output_dir, sys.argv)

    dbase = dataset.DBase(args.dataset)
    print(dbase)
    print('output directory', args.output_dir)
    print('output h5 file  ', args.h5file)

    print('Creating networks and loading parameters')
    detector = FaceDetector(detector=args.detector)
    print(detector)

    nrof_extracted_faces = 0
    nrof_unread_files = 0

    for i, cls in enumerate(dbase.classes):
        print('{}/{} {}'.format(i, dbase.nrof_classes, cls.name))

        # define output class directory
        output_class_dir = args.output_dir.joinpath(cls.name)
        ioutils.makedirs(output_class_dir)

        for image_path in cls.files:
            print(image_path)
            out_filename = output_class_dir.joinpath(Path(image_path).stem + '.png')

            try:
                # this function returns PIL.Image object
                img = ioutils.read_image(image_path)
            except (IOError, ValueError, IndexError) as e:
                nrof_unread_files += 1
                # print(e)
            else:
                boxes = detector.detect(img)
                nrof_faces = len(boxes)

                if nrof_faces == 0:
                    print('Unable to find face "{}"'.format(image_path))
                    continue

                if nrof_faces > 1 and args.detect_multiple_faces is False:
                    print('The number of detected faces more than one "{}"'.format(image_path))
                    continue

                nrof_extracted_faces += 1

                for i, box in enumerate(boxes):
                    output = image_processing(img, box, args.image_size, margin=args.margin)

                    out_filename_i = out_filename
                    if i > 0:
                        out_filename_i = out_filename.parent.joinpath('{}_{}{}'.format(out_filename.stem, i, out_filename.suffix))

                    ioutils.write_image(output, out_filename_i)
                    size = np.uint32((box.height, box.width))
                    h5utils.write(args.h5file, h5utils.filename2key(out_filename_i, 'size'), size)

    print(dbase)
    print('Number of successfully extracted faces: {}'.format(nrof_extracted_faces))
    print('The number of files that cannot be read', nrof_unread_files)


if __name__ == '__main__':
    main()
