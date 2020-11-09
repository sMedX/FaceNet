"""Validate a face recognizer.
"""
# MIT License
#
# Copyright (c) 2020 SMedX

import click
from pathlib import Path
import tensorflow as tf

from facenet import dataset, config, facenet, tfutils, ioutils, FaceNet
import numpy as np


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
    face_net = FaceNet(options['model'])

    for file in dbase.files:
        print(file)
        img1 = ioutils.read_image(file)
        array = ioutils.pil2array(img1)
        emb1 = face_net.image_to_embedding(array)

        img2 = img1.convert('L')
        array = ioutils.pil2array(img2)
        emb2 = face_net.image_to_embedding(array)

        diff = np.linalg.norm(emb1 - emb2)
        print(diff)


if __name__ == '__main__':
    main()
