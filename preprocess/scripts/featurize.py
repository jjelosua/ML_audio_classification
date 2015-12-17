# coding: utf-8
"""
Generate features from the wav numpy arrays
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import argrelmax
import librosa
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
OUTPUT_FILE = 'audio_features'
OUTPUT_PATH = os.path.abspath(os.path.join(cwd, '../../data/output/features'))
N_CORES = 7
# One second the length of a ring tone
SAMPLE_RATE = 8000
SMOOTH_W_LENGTH = 8000
SILENCE_CUTOFF = 0.015
RING_SEP_THRESHOLD = 2000
RING_AMP_DIFF_THRESHOLD = 0.02


def smooth(x, window_len=11, window='hanning'):
    '''smoothing function
    Can use any of these functions: flat, hanning, hamming, bartlett, blackman
    '''
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window function is not supported")
    s = np.r_[2*x[0]-x[window_len-1::-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len:-window_len+1]


def extract_chunks(data):
    """
    Extract silent and nonsilent chunks.
    """
    silence = data[0] == 0

    if silence:
        # Track [onset sample number, length] for each silent chunk
        silences = [[0, 0]]
        nonsilences = []
    else:
        # Track (onset sample number, amplitudes) for each nonsilent chunk
        nonsilences = [(0, [])]
        silences = []

    for i, amp in enumerate(data):
        if amp > 0:
            # Entering a new nonsilent chunk
            if silence:
                silence = False
                nonsilences.append((i, [amp]))
            else:
                nonsilences[-1][1].append(amp)
        else:
            # Entering a new silent chunk
            if not silence:
                silence = True
                silences.append([i, 1])
            else:
                silences[-1][1] += 1
    return silences, nonsilences


def get_ring_count_feature(data, lmax_indexes):
    '''analize the local maxima of the signal and extract
    ring count feature and last ring index'''
    number_rings = 0
    ring_ref_amp = 0
    last_ring_idx = 0

    local_maxima = data[lmax_indexes]
    # Take the time difference between each local maxima of the signal
    diff = np.diff(lmax_indexes, n=1)

    ring_candidates = np.where((diff >= (SAMPLE_RATE*5)-RING_SEP_THRESHOLD) &
                               (diff <= (SAMPLE_RATE*5)+RING_SEP_THRESHOLD))[0]
    for idx in ring_candidates:
        # Check differences between the local maximum amplitudes is small
        if not ring_ref_amp:
            ring_ref_amp = local_maxima[idx]
        amp_lower = (ring_ref_amp - (RING_AMP_DIFF_THRESHOLD))
        amp_upper = (ring_ref_amp + (RING_AMP_DIFF_THRESHOLD))
        if (amp_lower <= local_maxima[idx+1] <= amp_upper):
            number_rings += 1 if number_rings else 2
            last_ring_idx = lmax_indexes[idx+1]
    return (number_rings, last_ring_idx)


def get_local_maxima_idx(nonsilences):
    '''get the local maxima of the signal with a window of a second'''
    result = []
    for start_idx, amps in nonsilences:
        lmax_idx = start_idx + np.argmax(amps)
        result.append(lmax_idx)
    # result = argrelmax(data, order=4000)[0]
    return result


def get_percent_silence_feature(data):
    '''get the percentage of 0 amplitude signal'''
    result = len(np.where(data == 0)[0])/len(data)
    return result


def get_last_ring_to_end_feature(data, idx):
    '''get the ratio on how far from the end is the last ring'''
    result_samples = len(data) - idx if idx else len(data)
    return (result_samples / SAMPLE_RATE)


def preprocess(path):
    # load audio file with librosa
    data, sr = librosa.load(path, sr=None)
    # we are interested on the amplitude so translate the negative values
    data = np.abs(data)
    # Chop off anything below the cutoff pre-smoothing
    data[data < SILENCE_CUTOFF] = 0
    # Adjust window length if audio is really small
    window_len = min(SMOOTH_W_LENGTH, len(data))
    smooth_data = smooth(data, window='bartlett', window_len=window_len)

    # Chop off anything below the cutoff post-smoothing
    smooth_data[smooth_data < SILENCE_CUTOFF] = 0
    # rescale smooth data
    smooth_data *= 1.0/smooth_data.max()
    return smooth_data


def featurize(idx, row):
    try:
        # Preprocess audio into normalized and smoothed numpy array
        smooth_data = preprocess(row['audio_file'])

        silences, nonsilences = extract_chunks(smooth_data)

        # Compute percent_silence feature
        percent_silence = get_percent_silence_feature(smooth_data)

        # When no nonsilences have been detected set default values
        if not len(nonsilences[0][1]):
            ring_count = 0
            last_ring_to_end = len(smooth_data)
        else:
            # Compute local maxima
            lmax_idx = get_local_maxima_idx(nonsilences)
            # Compute ring_count feature
            ring_count, last_ring_idx = get_ring_count_feature(smooth_data,
                                                               lmax_idx)
            # Compute last_ring_to_end feature
            last_ring_to_end = get_last_ring_to_end_feature(smooth_data,
                                                            last_ring_idx)

        result = {
            'index': idx,
            'title': row['title'],
            'audio_file': row['audio_file'],
            'image_file': row['image_file'],
            'length': row['length'],
            'label': row['label'],
            'percent_silence': percent_silence,
            'ring_count': ring_count,
            'last_ring_to_end': last_ring_to_end,
        }
        return result
    except Exception as e:
        print('The audio {} raised an exception {}'.format(row['audio_file'],
                                                           e))


def process_audios(labeled=True):
    '''feature engineering for the audio file'''
    dataset = 'train' if labeled else 'test'
    data = pd.read_csv('{}/{}.csv'.format(INPUT_PATH, INPUT_FILES[dataset]))
    # data = data[data.length <= 300]
    # data_sample = data.sample(200)
    data_sample = data
    r = Parallel(n_jobs=N_CORES)(delayed(featurize)(j, row)
                                 for j, row in data_sample.iterrows())

    df_result = pd.DataFrame(r)
    df_result.to_csv('{}/{}/{}.csv'.format(OUTPUT_PATH, dataset, OUTPUT_FILE),
                     index=False,
                     columns=['index',
                              'title',
                              'audio_file',
                              'image_file',
                              'length',
                              'label',
                              'percent_silence',
                              'ring_count',
                              'last_ring_to_end'])
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
    process_audios(labeled=True)
    process_audios(labeled=False)


if __name__ == '__main__':
    run()
