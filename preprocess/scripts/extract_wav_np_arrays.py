# coding: utf-8
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
# Parallel execution libs
from joblib import Parallel, delayed
from collections import defaultdict
import joblib.parallel


# PARALLEL EXECUTION SETTINGS
# Override joblib callback default callback behavior
class CallBack(object):
    completed = defaultdict(int)

    def __init__(self, index, parallel):
        self.index = index
        self.parallel = parallel

    def __call__(self, index):
        CallBack.completed[self.parallel] += 1
        if CallBack.completed[self.parallel] % 100 == 0:
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
    'train': 'labeled_wav_local',
    'test': 'unlabeled_wav_local'
}
OUTPUT_FILE = 'wavs.npz'
OUTPUT_PATH = os.path.abspath(os.path.join(cwd, '../../data/output/npz'))
N_CORES = 7


def get_numpy_arr(idx, row):
    '''get the numpy array of the audio'''
    d = {}
    sample_rate, data = wavfile.read(row['audio_file'])
    # Normalize data causes broken pipe when executed in parallel
    # https://bugs.python.org/issue17560
    # data = data / (2.**15) if (data.dtype == 'int16') else data / (2.**31)
    d[str(idx)] = data
    return d


def process_audios(labeled=True):
    '''extract the image of the audio file'''
    dataset = 'train' if labeled else 'test'
    data = pd.read_csv('{}/{}.csv'.format(INPUT_PATH, INPUT_FILES[dataset]))
    data = data[data.length <= 300]
    # data_sample = data.sample(100)
    data_sample = data
    r = Parallel(n_jobs=N_CORES)(delayed(get_numpy_arr)(j, row)
                                 for j, row in data_sample.iterrows())
    result = {}
    # Transform to dictionary of dictionaries
    for item in r:
        for key, value in item.items():
            result[key] = value
    np.savez('{}/{}/{}'.format(OUTPUT_PATH, dataset, OUTPUT_FILE), **result)
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
