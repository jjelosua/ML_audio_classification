"""
Generate features from the wav numpy arrays
"""
import numpy as np
import pandas as pd
from time import time
from scipy.io import wavfile
from sup.progress import Progress
from collections import defaultdict
from joblib import Parallel, delayed

sample_rate = 8000
ring_var_thresh = 2000000
ring_min_length = sample_rate * 0.75 # ~6000 samples
ring_min_preceding_silence_length = 5000
ring_percent_samps_thresh = 0.7
ring_dev = 1500
n_jobs = 7

class CallBack():
    completed = defaultdict(int)

    def __init__(self, index, parallel):
        self.index = index
        self.parallel = parallel
        self.progress = Progress()
        self.total = 21100

    def __call__(self, index):
        CallBack.completed[self.parallel] += 1
        self.progress.print_progress(CallBack.completed[self.parallel]/self.total)
        if self.parallel._original_iterable:
            self.parallel.dispatch_next()

import joblib.parallel
joblib.parallel.CallBack = CallBack

def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]


def extract_chunks(smooth_data):
    """
    Extract silent and nonsilent chunks.
    """
    silence = smooth_data[0] == 0

    if silence:
        # Track [onset sample number, length] for each silent chunk
        silences = [[0, 0]]
        nonsilences = []
    else:
        # Track (onset sample number, amplitudes) for each nonsilent chunk
        nonsilences = [(0, [])]
        silences = []

    for i, amp in enumerate(smooth_data):
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


def nonsilence_vars(nonsilences):
    return [np.var(ns, ddof=1) for sample_num, ns in nonsilences]


def is_ring(chunk, variance, preceding_silence_length):
    # Estimates if a given nonsilent chunk is a ring
    #if preceding_silence_length < ring_min_preceding_silence_length:
        #return False

    if variance > ring_var_thresh:
        return False

    if len(chunk) < ring_min_length:
        return False

    if isinstance(chunk, list):
        chunk = np.array(chunk)

    med = np.median(chunk) # use median as it is a more resilient to outliers
    samps_within = chunk[np.where((chunk > med - ring_dev) & (chunk < med + ring_dev))]
    percent_samps_within = len(samps_within)/len(chunk)

    if percent_samps_within < ring_percent_samps_thresh:
        return False
    return True


def preprocess(arr, silence_cutoff=500, window_len=2000):
    # some arrs are in stereo, i.e. have two rows, convert to mono
    if len(arr.shape) > 1:
        arr = arr[:,0]
    arr = np.abs(arr)
    window_len = min(window_len, len(arr))
    smooth_arr = smooth(arr, window='hanning', window_len=window_len)

    # Chop off anything below the cutoff
    smooth_arr[smooth_arr < silence_cutoff] = 0
    return smooth_arr


def featurize_arr(row):
    wav = row['wav_path']
    sample_rate, arr = wavfile.read(wav)
    smooth_arr = preprocess(arr)

    silences, nonsilences = extract_chunks(smooth_arr)
    nonsilence_variances = nonsilence_vars(nonsilences)
    mean_nonsilence_variances = np.mean(nonsilence_variances)
    variance_of_nonsilence_variances = np.var(nonsilence_variances, ddof=1)

    # The preceding silence for a nonsilent chunk is the first silence that
    # comes before the nonsilent chunk
    preceding_silence_lengths = [next((length for spos, length in silences[::-1] if spos < pos), 0) for pos, ns in nonsilences]

    rings = [pos for (pos, ns), var, psl in zip(nonsilences, nonsilence_variances, preceding_silence_lengths) if is_ring(ns, var, psl)]
    ring_count = len(rings)

    if rings:
        last_ring_sample_number = rings[-1]
        last_ring_to_end_length = len(smooth_arr) - last_ring_sample_number
    else:
        last_ring_to_end_length = len(smooth_arr)

    seconds_length = len(smooth_arr)/sample_rate
    max_amplitude = np.max(smooth_arr)
    percent_silence = sum(s[1] for s in silences)/len(smooth_arr)
    mean_silence_length = np.mean([s[1] for s in silences])

    feats = {
        'mean_silence_length': mean_silence_length,
        'percent_silence': percent_silence,
        'ring_count': ring_count,
        'max_amplitude': max_amplitude,
        'seconds_length': seconds_length,
        'mean_nonsilence_variances': mean_nonsilence_variances,
        'variance_of_nonsilence_variances': variance_of_nonsilence_variances,
        'last_ring_to_end_length': last_ring_to_end_length,
    }
    return feats


if __name__ == '__main__':
    data = pd.read_csv('data.csv')

    s = time()
    print('Featurizing...')
    features = Parallel(n_jobs=n_jobs)(delayed(featurize_arr)(row) for i, row in data.iterrows())
    print('Took {:.2f}s'.format(time() - s))

    print('Creating dataframe...')
    f = pd.DataFrame(features)
    df = pd.concat([data, f], axis=1)
    df.to_csv('featurized.csv')
    print('Done')
