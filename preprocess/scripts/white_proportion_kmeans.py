# coding: utf-8
import os
import numpy as np
import pandas as pd
import cv2
# Parallel execution libs
from joblib import Parallel, delayed
from collections import defaultdict
import joblib.parallel

# cluster the pixel intensities
from sklearn.cluster import KMeans
clt = KMeans(n_clusters=3)


# PARALLEL EXECUTION SETTINGS
# Override joblib callback default callback behavior
class CallBack(object):
    completed = defaultdict(int)

    def __init__(self, index, parallel):
        self.index = index
        self.parallel = parallel

    def __call__(self, index):
        CallBack.completed[self.parallel] += 1
        if CallBack.completed[self.parallel] % 10 == 0:
            print("processed {} items"
                  .format(CallBack.completed[self.parallel]))
        if self.parallel._original_iterable:
            self.parallel.dispatch_next()
# MonkeyPatch Callback
joblib.parallel.CallBack = CallBack


# GLOBAL SETTINGS
cwd = os.path.dirname(__file__)
INPUT_PATH = os.path.join(cwd, '../../data/input')
INPUT_FILES = {
    'train': 'labeled_wav_image_local',
    'test': 'unlabeled_wav_image_local'
}
OUTPUT_PATH = os.path.abspath(os.path.join(cwd, '../../data/output/features'))
OUTPUT_FILE = 'image_features'
N_CORES = 7


# Helper functions
# Get the proportion of colors based on KMeans
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    hist.sort()
    hist = hist[::-1]
    # return the histogram
    return hist


def white_ratio_feature(index, row):
    result = {
        'index': index,
        'label': row['label'],
        'white_proportion': None,
    }
    image = cv2.imread(row['image_file'])
    if image is not None:
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt.fit(image)
        hist = centroid_histogram(clt)
        result['white_proportion'] = hist[0]
    return result


def process_audios(labeled=True):
    '''extract the image of the audio file'''
    dataset = 'train' if labeled else 'test'
    data = pd.read_csv('{}/{}.csv'.format(INPUT_PATH, INPUT_FILES[dataset]))
    data = data[data.length <= 300]
    # data_sample = data.sample(100)
    data_sample = data
    r = Parallel(n_jobs=N_CORES)(delayed(white_ratio_feature)(j, row)
                                 for j, row in data_sample.iterrows())

    df_result = pd.DataFrame(r)
    df_result.to_csv('{}/{}/{}.csv'.format(OUTPUT_PATH, dataset, OUTPUT_FILE),
                     index=False,
                     columns=['index', 'label', 'white_proportion'])
    print('finished processing {}.csv'.format(INPUT_FILES[dataset]))


def create_folders(paths=None):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def create_folder_structure():
    '''Create a list of the required folders for the script execution'''
    paths = [
        # Training set audio files
        '%s/train' % OUTPUT_PATH,
        # Test set audio files
        '%s/test' % OUTPUT_PATH]
    create_folders(paths)


def run():
    # Initialize execution context
    create_folder_structure()
    # Download training set
    # Download test set
    process_audios(labeled=True)
    process_audios(labeled=False)


if __name__ == '__main__':
    run()
