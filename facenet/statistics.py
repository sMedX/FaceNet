
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
        self.tn_rates = None

        self.embeddings = split_embeddings(embeddings, labels)
        self.distances = []

        for i, emb1 in enumerate(self.embeddings):
            self.distances.append([])

            for k, emb2 in enumerate(self.embeddings[:i]):
                self.distances[i].append(pairwise_distances(emb1, emb2, metric=metric))

            self.distances[i].append(pairwise_distances(emb1, metric=metric))

    def compute(self, thresholds):

        if isinstance(thresholds, Iterable) is False:
            thresholds = np.array([thresholds])

        tp = np.zeros(thresholds.size)
        tn = np.zeros(thresholds.size)
        fp = np.zeros(thresholds.size)
        fn = np.zeros(thresholds.size)

        for i, distances_i in enumerate(self.distances):
            for k, distances_k in enumerate(distances_i):
                for n, threshold in enumerate(thresholds):
                    count = np.count_nonzero(distances_k < threshold)

                    if i == k:
                        tp[n] += count
                        fn[n] += distances_k.size - count
                    else:
                        fp[n] += count
                        tn[n] += distances_k.size - count

        self.accuracy = (tp + tn) / (tp + fp + tn + fn)

        # precision
        i = (tp + fp) > 0
        self.precision = np.ones(thresholds.size)
        self.precision[i] = tp[i] / (tp[i] + fp[i])

        # true positive rate, validation rate, sensitivity or recall
        i = (tp + fn) > 0
        self.tp_rates = np.ones(thresholds.size)
        self.tp_rates[i] = tp[i] / (tp[i] + fn[i])

        # true negative rate, 1 - false alarm rate, specificity
        i = (tn + fp) > 0
        self.tn_rates = np.ones(thresholds.size)
        self.tn_rates[i] = tn[i] / (tn[i] + fp[i])

    @property
    def fp_rates(self):
        # false positive rate, false alarm rate
        return 1 - self.tn_rates

    @property
    def fn_rates(self):
        # false negative rate,
        return 1 - self.tp_rates


class Validation:
    def __init__(self, thresholds, embeddings, labels,
                 far_target=1e-3, nrof_folds=10,
                 metric=0, subtract_mean=False):
        """

        :param thresholds:
        :param embeddings:
        :param labels:
        :param far_target: target false alarm rate (face pairs that was incorrectly classified as the same)
        :param nrof_folds:
        :param metric:
        :param subtract_mean:
        """

        self.subtract_mean = subtract_mean
        self.metric = metric
        self.far_target = far_target

        self.labels = np.array(labels)
        self.embeddings = embeddings
        assert (embeddings.shape[0] == len(labels))

        self.best_thresholds = np.zeros(nrof_folds)
        self.far_thresholds = np.zeros(nrof_folds)
        self.accuracy = np.zeros(nrof_folds)
        self.precision = np.zeros(nrof_folds)
        self.tp_rates = np.zeros(nrof_folds)
        self.tn_rates = np.zeros(nrof_folds)

        self.tp_rate_far_target = np.zeros(nrof_folds)
        self.tn_rate_far_target = np.zeros(nrof_folds)

        tp_rates = np.zeros((nrof_folds, len(thresholds)))
        tn_rates = np.zeros((nrof_folds, len(thresholds)))

        k_fold = KFold(n_splits=nrof_folds, shuffle=False)
        indices = np.arange(len(labels))

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            print('\rvalidation {}/{}'.format(fold_idx, nrof_folds), end=utils.end(fold_idx, nrof_folds))

            # evaluations with train set and define the best threshold for the fold
            conf_matrix = ConfidenceMatrix(self.embeddings[train_set], self.labels[train_set], metric=self.metric)
            conf_matrix.compute(thresholds)
            tp_rates[fold_idx, :] = conf_matrix.tp_rates
            tn_rates[fold_idx, :] = conf_matrix.tn_rates

            self.best_thresholds[fold_idx] = thresholds[np.argmax(conf_matrix.accuracy)]

            # find the threshold that gives FAR (FPR, 1-TNR) = far_target
            if np.max(conf_matrix.fp_rates) >= self.far_target:
                f = interpolate.interp1d(conf_matrix.fp_rates, thresholds, kind='slinear')
                self.far_thresholds[fold_idx] = f(self.far_target)
            else:
                self.far_thresholds[fold_idx] = 0.0

            # evaluations with test set
            conf_matrix = ConfidenceMatrix(self.embeddings[test_set], self.labels[test_set], metric=self.metric)
            conf_matrix.compute(self.best_thresholds[fold_idx])
            self.accuracy[fold_idx] = conf_matrix.accuracy
            self.precision[fold_idx] = conf_matrix.precision
            self.tp_rates[fold_idx] = conf_matrix.tp_rates
            self.tn_rates[fold_idx] = conf_matrix.tn_rates

            conf_matrix.compute(self.far_thresholds[fold_idx])
            self.tp_rate_far_target[fold_idx] = conf_matrix.tp_rates
            self.tn_rate_far_target[fold_idx] = conf_matrix.tn_rates

        # compute area under curve and equal error rate
        tp_rates = np.mean(tp_rates, axis=0)
        tn_rates = np.mean(tn_rates, axis=0)

        try:
            self.auc = sklearn.metrics.auc(1 - tn_rates, tp_rates)
        except Exception:
            self.auc = -1

        try:
            self.eer = brentq(lambda x: 1. - x - interpolate.interp1d(1 - tn_rates, tp_rates)(x), 0., 1.)
        except Exception:
            self.eer = -1

        print('\nValidation report')
        print('Accuracy:  {:2.5f}+-{:2.5f}'.format(np.mean(self.accuracy), np.std(self.accuracy)))
        print('Precision: {:2.5f}+-{:2.5f}'.format(np.mean(self.precision), np.std(self.precision)))
        print('Sensitivity (TPR, 1-a type 1 error): {:2.5f}+-{:2.5f}'.format(np.mean(self.tp_rates), np.std(self.tp_rates)))
        print('Specificity (TNR, 1-b type 2 error): {:2.5f}+-{:2.5f}'.format(np.mean(self.tn_rates), np.std(self.tn_rates)))
        print()
        print('Area Under Curve (AUC): {:1.5f}'.format(self.auc))
        print('Equal Error Rate (EER): {:1.5f}'.format(self.eer))
        print('Threshold: {:2.5f}+-{:2.5f}'.format(np.mean(self.best_thresholds), np.std(self.best_thresholds)))
        print()

    def write_report(self, elapsed_time, args, file=None, dbase_info=None):
        if file is None:
            dir_name = pathlib.Path(args.model).expanduser()
            if dir_name.is_file():
                dir_name = dir_name.parent
            file = dir_name.joinpath('report.txt')
        else:
            file = pathlib.Path(file).expanduser()

        git_hash, git_diff = utils.git_hash()
        with file.open('at') as f:
            f.write('{}\n'.format(datetime.datetime.now()))
            f.write('git hash: {}\n'.format(git_hash))
            f.write('git diff: {}\n'.format(git_diff))
            f.write('{}'.format(dbase_info))
            f.write('model: {}\n'.format(args.model))
            f.write('embedding size: {}\n'. format(self.embeddings.shape[1]))
            f.write('elapsed time: {}\n'.format(elapsed_time))
            f.write('time per image: {}\n'.format(elapsed_time/self.embeddings.shape[0]))
            f.write('distance metric: {}\n'.format(self.metric))
            f.write('subtract mean: {}\n'.format(self.subtract_mean))
            f.write('\n')
            f.write('Criterion of maximum accuracy\n')
            f.write('Accuracy:  {:2.5f}+-{:2.5f}\n'.format(np.mean(self.accuracy), np.std(self.accuracy)))
            f.write('Precision: {:2.5f}+-{:2.5f}\n'.format(np.mean(self.precision), np.std(self.precision)))
            f.write('Sensitivity (TPR): {:2.5f}+-{:2.5f}\n'.format(np.mean(self.tp_rates), np.std(self.tp_rates)))
            f.write('Specificity (TNR): {:2.5f}+-{:2.5f}\n'.format(np.mean(self.tn_rates), np.mean(self.tn_rates)))
            f.write('\n')
            f.write('Area Under Curve (AUC): {:1.5f}\n'.format(self.auc))
            f.write('Equal Error Rate (EER): {:1.5f}\n'.format(self.eer))
            f.write('Threshold: {:2.5f}+-{:2.5f}\n'.format(np.mean(self.best_thresholds), np.std(self.best_thresholds)))
            f.write('\n')
            f.write('Criterion of false alarm rate target: (FPR, 1 - TNR): {:2.5f}\n'.format(self.far_target))
            f.write('Sensitivity (TPR, 1-a type 1 error): {:2.5f}+-{:2.5f}\n'.
                    format(np.mean(self.tp_rate_far_target), np.std(self.tp_rate_far_target)))
            f.write('Specificity (TNR, 1-b type 2 error): {:2.5f}+-{:2.5f}\n'.
                    format(np.mean(self.tn_rate_far_target), np.std(self.tn_rate_far_target)))
            f.write('Threshold: {:2.5f}+-{:2.5f}\n'.format(np.mean(self.far_thresholds), np.std(self.far_thresholds)))
            f.write('\n')

        print('Report has been printed to the file: {}'.format(file))


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
