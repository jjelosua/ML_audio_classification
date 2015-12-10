# coding: utf-8
import os
import numpy as np
import pandas as pd
import cv2
# Parallel execution libs
from joblib import Parallel, delayed
from collections import defaultdict
import joblib.parallel


# PARALLEL EXECUTION SETTINGS
# Override joblib callback default callback behavior
class BatchCompletionCallBack(object):
    completed = defaultdict(int)

    def __init__(self, dispatch_timestamp, batch_size, parallel):
        self.dispatch_timestamp = dispatch_timestamp
        self.batch_size = batch_size
        self.parallel = parallel

    def __call__(self, out):
        BatchCompletionCallBack.completed[self.parallel] += 1
        if BatchCompletionCallBack.completed[self.parallel] % 10 == 0:
            print("processed {} items"
                  .format(BatchCompletionCallBack.completed[self.parallel]))
        if self.parallel._original_iterator is not None:
            self.parallel.dispatch_next()
# MonkeyPatch BatchCompletionCallBack
joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack


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


def white_ratio_feature(index, row):
    result = {
        'index': index,
        'label': row['label'],
        'white_proportion': None,
    }
    # Read image in grayscale
    im_gray = cv2.imread(row['image_file'], 0)
    if im_gray is not None:
        unique, counts = np.unique(im_gray, return_counts=True)
        # White 255 divided by all
        white_proportion = counts[-1]/sum(counts)
        result['white_proportion'] = white_proportion
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
