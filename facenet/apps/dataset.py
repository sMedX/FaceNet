# coding:utf-8
"""Application to print information about dataset
"""
# MIT License
# 
# Copyright (c) 2020 sMedX
# 
import click
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from facenet import dataset, config


@click.command()
@click.option('--path', default=config.default_train_dataset, type=Path,
              help='Path to dataset directory to check.')
def main(**options):

    dbase = dataset.DBase(dataset.DefaultConfig(options['path']))
    print(dbase)

    for f in tqdm(dbase.files):
        try:
            image = Image.open(f)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()

