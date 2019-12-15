
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys

import os
import datetime
import numpy as np
from skimage import io
import sklearn
from sklearn.model_selection import KFold
from scipy import spatial, interpolate
from scipy.optimize import brentq
from collections.abc import Iterable
import math
import pathlib

from facenet import utils


def pairwise_distances(xa, xb=None, metric=0):
    if metric == 0:
        # squared Euclidian distance
        if xb is None:
            dist = spatial.distance.pdist(xa, metric='sqeuclidean')
        else:
            dist = spatial.distance.cdist(xa, xb, metric='sqeuclidean')
    elif metric == 1:
        # distance based on cosine similarity
        if xb is None:
            dist = spatial.distance.pdist(xa, metric='cosine')
        else:
            dist = spatial.distance.cdist(xa, xb, metric='cosine')
        dist = np.arccos(1 - dist) / math.pi
    else:
        raise 'Undefined distance metric %d' % metric

    return dist


def precision(tp, fp):
    p = np.ones(len(tp))
    i = (tp + fp) > 0
    p[i] = tp[i] / (tp[i] + fp[i])
    return p


def split_embeddings(embeddings, labels):
    emb_list = []
    for label in np.unique(labels):
        emb_array = embeddings[labels == label]
        emb_list.append(emb_array)
    return emb_list


class ConfidenceMatrix:
    def __init__(self, embeddings, labels, metric=0):
        self.precision = None
        self.accuracy = None
        self.tp_rates = None
        self.fp_rates = None

        self.embeddings = split_embeddings(embeddings, labels)
        self.distances = [[] for _ in range(len(self.embeddings))]

        for i, emb1 in enumerate(self.embeddings):
            for k, emb2 in enumerate(self.embeddings[:i]):
                self.distances[i].append(pairwise_distances(emb1, emb2, metric=metric))
            self.distances[i].append(pairwise_distances(emb1, metric=metric))

    def compute(self, thresholds):

        if isinstance(thresholds, Iterable) is False:
            thresholds = np.array([thresholds])

        tp = np.zeros(thresholds.size, dtype=int)
        tn = np.zeros(thresholds.size, dtype=int)
        fp = np.zeros(thresholds.size, dtype=int)
        fn = np.zeros(thresholds.size, dtype=int)

        for i in range(len(self.distances)):
            for k in range(len(self.distances[i])):
                for n, threshold in enumerate(thresholds):
                    count = np.sum(self.distances[i][k][:] < threshold)

                    if i == k:
                        tp[n] += count
                        fn[n] += self.distances[i][k].size - count
                    else:
                        fp[n] += count
                        tn[n] += self.distances[i][k].size - count

        self.precision = precision(tp, fp)
        self.accuracy = (tp + tn) / (tp + fp + tn + fn)

        # true positive rate, validation rate, sensitivity or recall
        self.tp_rates = tp / (tp + fn)

        # false positive rate, false alarm rate, 1 - specificity
        self.fp_rates = fp / (fp + tn)


class Validation:
    def __init__(self, thresholds, embeddings, dbase,
                 far_target=1e-3, nrof_folds=10,
                 distance_metric=0, subtract_mean=False):

        self.dbase = dbase
        self.subtract_mean = subtract_mean
        self.distance_metric = distance_metric
        labels = np.array(dbase.labels)

        self.embeddings = embeddings
        assert (embeddings.shape[0] == len(labels))

        self.best_thresholds = np.zeros(nrof_folds)
        self.accuracy = np.zeros(nrof_folds)
        self.precision = np.zeros(nrof_folds)
        self.tp_rates = np.zeros(nrof_folds)
        self.fp_rates = np.zeros(nrof_folds)

        tp_rates = np.zeros((nrof_folds, len(thresholds)))
        fp_rates = np.zeros((nrof_folds, len(thresholds)))

        k_fold = KFold(n_splits=nrof_folds, shuffle=False)
        indices = np.arange(len(labels))

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            print('\rvalidation {}/{}'.format(fold_idx, nrof_folds), end=utils.end(fold_idx, nrof_folds))

            # evaluations with train set and define the best threshold for the fold
            conf_matrix = ConfidenceMatrix(embeddings[train_set], labels[train_set])
            conf_matrix.compute(thresholds)
            self.best_thresholds[fold_idx] = thresholds[np.argmax(conf_matrix.accuracy)]

            # evaluations with test set
            conf_matrix = ConfidenceMatrix(embeddings[test_set], labels[test_set])
            conf_matrix.compute(self.best_thresholds[fold_idx])
            self.accuracy[fold_idx] = conf_matrix.accuracy
            self.precision[fold_idx] = conf_matrix.precision
            self.tp_rates[fold_idx] = conf_matrix.tp_rates
            self.fp_rates[fold_idx] = conf_matrix.fp_rates

            conf_matrix = ConfidenceMatrix(embeddings[test_set], labels[test_set])
            conf_matrix.compute(thresholds)
            tp_rates[fold_idx, :] = conf_matrix.tp_rates
            fp_rates[fold_idx, :] = conf_matrix.fp_rates

        # accuracy
        self.accuracy_mean = np.mean(self.accuracy)
        self.accuracy_std = np.std(self.accuracy)

        # precision
        self.precision_mean = np.mean(self.precision)
        self.precision_std = np.std(self.precision)

        self.tp_rates_mean = np.mean(self.tp_rates)
        self.tp_rates_std = np.std(self.tp_rates)

        self.fp_rates_mean = np.mean(self.fp_rates)
        self.fp_rates_std = np.std(self.fp_rates)

        self.best_threshold = np.mean(self.best_thresholds)
        self.best_threshold_std = np.std(self.best_thresholds)

        # compute area under curve and equal error rate
        tp_rates = np.mean(tp_rates, axis=0)
        fp_rates = np.mean(fp_rates, axis=0)

        try:
            self.auc = sklearn.metrics.auc(fp_rates, tp_rates)
        except Exception:
            self.auc = -1

        try:
            self.eer = brentq(lambda x: 1. - x - interpolate.interp1d(fp_rates, tp_rates)(x), 0., 1.)
        except Exception:
            self.eer = -1

    def print(self):
        print('Accuracy: {:2.5f}+-{:2.5f}'.format(self.accuracy_mean, self.accuracy_std))
        print('Area Under Curve (AUC): {:1.5f}'.format(self.auc))
        print('Equal Error Rate (EER): {:1.5f}'.format(self.eer))
        print('Threshold: {:2.5f}+-{:2.5f}'.format(self.best_threshold, self.best_threshold_std))

    def write_report(self, elapsed_time, args, file=None):
        if file is None:
            dir_name = pathlib.Path(args.model).expanduser()
            if dir_name.is_file():
                dir_name = dir_name.parent
            file = dir_name.joinpath('report.txt')
        else:
            file = pathlib.Path(file).expanduser()

        print('Report has been printed to the file: {}'.format(file))

        git_hash, git_diff = utils.git_hash()
        with file.open('at') as f:
            f.write('{}\n'.format(datetime.datetime.now()))
            f.write('git hash: {}\n'.format(git_hash))
            f.write('git diff: {}\n'.format(git_diff))
            f.write('model: {}\n'.format(args.model))
            f.write('embedding size: {}\n'. format(self.embeddings.shape[1]))
            f.write('elapsed time: {}\n'.format(elapsed_time))
            f.write('time per image: {}\n'.format(elapsed_time/self.embeddings.shape[0]))
            f.write('data set: {}\n'.format(self.dbase.config.path))
            f.write('h5 file: {}\n'.format(self.dbase.config.h5file))
            f.write('portion of images: {}\n'.format(self.dbase.config.portion))
            f.write('number of folders {}\n'.format(self.dbase.nrof_classes))
            f.write('numbers of images {} and pairs {}\n'.format(self.dbase.nrof_images, self.dbase.nrof_pairs))
            f.write('distance metric: {}\n'.format(self.distance_metric))
            f.write('subtract mean: {}\n'.format(self.subtract_mean))
            f.write('\n')
            f.write('Accuracy: {:2.5f}+-{:2.5f}\n'.format(self.accuracy_mean, self.accuracy_std))
            f.write('Precision: {:2.5f}+-{:2.5f}\n'.format(self.precision_mean, self.precision_std))
            f.write('Sensitivity (TPR): {:2.5f}+-{:2.5f}\n'.format(self.tp_rates_mean, self.tp_rates_std))
            f.write('Specificity (TNR): {:2.5f}+-{:2.5f}\n'.format(1-self.fp_rates_mean, self.fp_rates_std))
            f.write('\n')
            f.write('Area Under Curve (AUC): {:1.5f}\n'.format(self.auc))
            f.write('Equal Error Rate (EER): {:1.5f}\n'.format(self.eer))
            f.write('Threshold: {:2.5f}+-{:2.5f}\n'.format(self.best_threshold, self.best_threshold_std))
            f.write('\n')


class FalseExamples:
    def __init__(self, dbase, tfrecord, threshold, metric=0, subtract_mean=False):
        self.dbase = dbase
        self.embeddings = tfrecord.embeddings
        self.threshold = threshold
        self.metric = metric
        self.subtract_mean = subtract_mean

    def write_false_pairs(self, fpos_dir, fneg_dir, nrof_fpos_images=10, nrof_fneg_images=2):

        if not os.path.isdir(fpos_dir):
            os.makedirs(fpos_dir)

        if not os.path.isdir(fneg_dir):
            os.makedirs(fneg_dir)

        if self.subtract_mean:
            mean = np.mean(self.embeddings, axis=0)
        else:
            mean = 0

        for folder1 in range(self.dbase.nrof_folders):
            print('\rWrite false examples {}/{}'.format(folder1, self.dbase.nrof_folders),
                  end=utils.end(folder1, self.dbase.nrof_folders))

            files1, embeddings1 = self.dbase.extract_data(folder1, self.embeddings)

            # search false negative pairs
            distances = pairwise_distances(embeddings1 - mean, metric=self.metric)
            distances = spatial.distance.squareform(distances)

            for n in range(nrof_fpos_images):
                # find maximal distances
                i, k = np.unravel_index(np.argmax(distances), distances.shape)

                if distances[i, k] > self.threshold:
                    self.write_image(distances[i, k], files1[i], files1[k], fneg_dir)
                    distances[[i, k], :] = -1
                    distances[:, [i, k]] = -1
                else:
                    break

            # search false positive pairs
            for folder2 in range(folder1+1, self.dbase.nrof_folders):
                files2, embeddings2 = self.dbase.extract_data(folder2, self.embeddings)

                distances = pairwise_distances(embeddings1-mean, embeddings2-mean, metric=self.metric)

                for n in range(nrof_fneg_images):
                    # find minimal distances
                    i, k = np.unravel_index(np.argmin(distances), distances.shape)

                    if distances[i, k] < self.threshold:
                        self.write_image(distances[i, k], files1[i], files2[k], fpos_dir)
                        distances[i, :] = np.Inf
                        distances[:, k] = np.Inf
                    else:
                        break

    def generate_filename(self, dirname, distance, file1, file2):
        dir1 = os.path.basename(os.path.dirname(file1))
        name1 = os.path.splitext(os.path.basename(file1))[0]

        dir2 = os.path.basename(os.path.dirname(file2))
        name2 = os.path.splitext(os.path.basename(file2))[0]

        return os.path.join(dirname, '{:2.3f} & {}|{} & {}|{}.png'.format(distance, dir1, name1, dir2, name2))

    def generate_text(self, distance, file1, file2):

        def text(file):
            return os.path.join(os.path.basename(os.path.dirname(file)), os.path.splitext(os.path.basename(file))[0])

        return '{} & {}\n{:2.3f}/{:2.3f}'.format(text(file1), text(file2), distance, self.threshold)

    def write_image(self, distance, file1, file2, dirname, fsize=13):
        fname = self.generate_filename(dirname, distance, file1, file2)
        text = self.generate_text(distance, file1, file2)

        img1 = io.imread(file1)
        img2 = io.imread(file2)
        img = Image.fromarray(np.concatenate([img1, img2], axis=1))

        if sys.platform == 'win32':
            font = ImageFont.truetype("arial.ttf", fsize)
        else:
            font = ImageFont.truetype("LiberationSans-Regular.ttf", fsize)

        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, (0, 255, 0), font=font)

        img.save(fname)
