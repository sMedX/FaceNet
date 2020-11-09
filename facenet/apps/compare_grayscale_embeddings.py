"""Validate a face recognizer.
"""
# MIT License
#
# Copyright (c) 2020 SMedX

import click
from pathlib import Path

import numpy as np

from facenet import dataset, config, ioutils, DefaultConfig, FaceNet


@click.command()
@click.option('--path', default=config.faces_dir,
              help='Path to directory with extracted faces to evaluate embeddings.')
@click.option('--model', default=config.default_model, type=Path,
              help='Path to directory with model.')
def main(**options):

    # read dataset
    dbase = dataset.DBase(options['path'])
    print(dbase)

    # create model
    conf = DefaultConfig(path=options['model'],
                         output='embeddings:0')     # 'InceptionResnetV1/Bottleneck/BatchNorm/Reshape_1:0')
    face_net = FaceNet(conf)

    for file in dbase.files:
        img1 = ioutils.read_image(file)
        arr1 = ioutils.pil2array(img1)
        emb1 = face_net.image_to_embedding(arr1)

        img2 = img1.convert('L')
        arr2 = ioutils.pil2array(img2)
        emb2 = face_net.image_to_embedding(arr2)

        diff1 = 2*np.linalg.norm(emb1 - emb2)**2/(np.linalg.norm(emb1)**2 + np.linalg.norm(emb2)**2)
        diff2 = np.abs(np.linalg.norm(emb1)**2 - np.linalg.norm(emb2)**2) / \
                      (np.linalg.norm(emb1)**2 + np.linalg.norm(emb2)**2)

        file = Path(file)
        print(Path(file.parent.name, file.name), diff1, diff2)


if __name__ == '__main__':
    main()
