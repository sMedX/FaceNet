"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# Copyright (c) 2020 Ruslan N. Kosarev

import sys
import click
from pathlib import Path
import numpy as np
from datetime import datetime

from facenet import dataset, ioutils, h5utils
from facenet.detectors.face_detector import image_processing, FaceDetector
from facenet import facenet, config


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options of the application.')
def main(**args_):
    args = config.YAMLConfig(args_['config'])

    if args.outdir is None:
        args.outdir = '{}_{}_extracted_{}'.format(Path(args.dataset.path), args.detector, args.image_size)
    args.outdir = Path(args.outdir).expanduser()
    ioutils.makedirs(args.outdir)

    if args.h5file is None:
        args.h5file = args.outdir.joinpath('statistics.h5')
    args.h5file = Path(args.h5file).expanduser()

    # store some git revision info in a text file in the log directory
    facenet.store_revision_info(Path(__file__).parent, args.outdir, ' '.join(sys.argv))

    # write arguments and store some git revision info in a text files in the log directory
    ioutils.write_arguments(args, args.outdir.joinpath('arguments.yaml'))
    ioutils.store_revision_info(args.outdir, sys.argv)

    dbase = dataset.DBase(args.dataset)
    print(dbase)
    print('output directory', args.outdir)
    print('output h5 file  ', args.h5file)

    print('Creating networks and loading parameters')
    detector = FaceDetector(detector=args.detector)
    print(detector)

    nrof_extracted_faces = 0
    nrof_unread_files = 0

    for i, cls in enumerate(dbase.classes):

        # define output class directory
        output_class_dir = args.outdir.joinpath(cls.name)
        ioutils.makedirs(output_class_dir)

        for k, image_path in enumerate(cls.files):
            print('{}/{} {}/{} {}'.format(i, dbase.nrof_classes, k, cls.nrof_images, image_path))
            out_filename = output_class_dir.joinpath(Path(image_path).stem + '.png')

            try:
                # this function returns PIL.Image object
                img = ioutils.read_image(image_path)
            except (IOError, ValueError, IndexError) as e:
                nrof_unread_files += 1
                print(e)
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

                for n, box in enumerate(boxes):
                    output = image_processing(img, box, args.image_size, margin=args.margin)

                    out_filename_n = out_filename
                    if n > 0:
                        out_filename_n = out_filename.parent.joinpath('{}_{}{}'.format(out_filename.stem, n, out_filename.suffix))

                    ioutils.write_image(output, out_filename_n)
                    size = np.uint32((box.height, box.width))
                    h5utils.write(args.h5file, h5utils.filename2key(out_filename_n, 'size'), size)

    print(dbase)
    print('Number of files that cannot be read', nrof_unread_files)
    print('Number of extracted faces', nrof_extracted_faces)

    report_file = args.outdir.joinpath('report.txt')
    with Path(report_file).open('w') as f:
        f.write('{}\n'.format(datetime.now()))
        f.write('{}\n'.format(dbase.__repr__()))
        f.write('\n')
        f.write('Number of files that cannot be read {}\n'.format(nrof_unread_files))
        f.write('Number of extracted faces {}\n'.format(nrof_extracted_faces))


if __name__ == '__main__':
    main()
