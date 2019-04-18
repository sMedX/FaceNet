
import os
import numpy as np


def get_files(dirname):
    list_of_files = []

    for root, dirs, files in os.walk(dirname):
        list_of_files += [os.path.join(root, file) for file in files]

    return list_of_files


def label_matrix(image_paths, diagonal=True):

    basenames = [os.path.basename(os.path.dirname(path)) for path in image_paths]

    labels = np.zeros([len(image_paths), len(image_paths)], dtype=np.uint8)

    for i, basename1 in enumerate(basenames):
        for k, basename2 in enumerate(basenames[:i]):
            if basename1 == basename2:
                labels[i][k] = labels[k][i] = 1

        if diagonal:
            labels[i][i] = 1

    return labels

