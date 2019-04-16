
import os


def get_files(dirname):
    list_of_files = []

    for root, dirs, files in os.walk(dirname):
        list_of_files += [os.path.join(root, file) for file in files]

    return list_of_files
