"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# Copyright (c) 2020 Ruslan N. Kosarev

import click
from pathlib import Path
import numpy as np
from tqdm import tqdm

from facenet import dataset, ioutils, h5utils
from facenet.detectors.face_detector import image_processing, FaceDetector
from facenet import config


@click.command()
@click.option('--config', default=config.default_app_config(__file__), type=Path,
              help='Path to yaml config file with used options of the application.')
def main(**options):
    options = config.ExtractFaces(options)

    dbase = dataset.DBase(options.dataset)
    ioutils.write_text_log(options.logfile, dbase)
    print('input dataset:', dbase)

    print('output directory', options.outdir)
    print('output h5 file  ', options.h5file)

    print('Creating networks and loading parameters')
    detector = FaceDetector(detector=options.detector)
    ioutils.write_text_log(options.logfile, detector)
    print(detector)

    nrof_extracted_faces = 0
    nrof_unread_files = 0

    with tqdm(total=dbase.nrof_classes) as bar:
        for i, cls in enumerate(dbase.classes):

            # define output class directory
            output_class_dir = options.outdir.joinpath(cls.name)
            ioutils.makedirs(output_class_dir)

            for k, image_path in enumerate(cls.files):
                bar.set_postfix_str('{}'.format('[{}/{}] {}'.format(k, cls.nrof_images, str(cls))))
                out_filename = output_class_dir.joinpath(Path(image_path).stem + '.png')

                try:
                    # this function returns PIL.Image object
                    img = ioutils.read_image(image_path)
                    img_array = ioutils.pil2array(img, mode=detector.mode)
                except Exception as e:
                    nrof_unread_files += 1
                    # print(e)
                else:
                    boxes = detector.detect(img_array)
                    nrof_faces = len(boxes)

                    if nrof_faces == 0:
                        # print('Unable to find face "{}"'.format(image_path))
                        continue

                    if nrof_faces > 1 and options.detect_multiple_faces is False:
                        # print('The number of detected faces more than one "{}"'.format(image_path))
                        continue

                    nrof_extracted_faces += 1

                    for n, box in enumerate(boxes):
                        output = image_processing(img, box, options.image)

                        out_filename_n = out_filename
                        if n > 0:
                            out_filename_n = out_filename.parent.joinpath('{}_{}{}'.format(out_filename.stem, n, out_filename.suffix))

                        ioutils.write_image(output, out_filename_n)
                        size = np.uint32((box.height, box.width))
                        h5utils.write(options.h5file, h5utils.filename2key(out_filename_n, 'size'), size)
            bar.update()

    out_dbase = dataset.DBase(dataset.DefaultConfig(options.outdir))
    ioutils.write_text_log(options.logfile, out_dbase)

    ioutils.write_text_log(options.logfile, f'Number of files that cannot be read {nrof_unread_files}')
    ioutils.write_text_log(options.logfile, f'Number of extracted faces {nrof_extracted_faces}')

    print('\n')
    print('Number of extracted faces', nrof_extracted_faces)
    print('Logs have been written to the file', options.logfile)


if __name__ == '__main__':
    main()
