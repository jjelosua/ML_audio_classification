# coding: utf-8
import os
import re
import subprocess
from csvkit.py3 import CSVKitDictReader, CSVKitDictWriter
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
    'train': 'labeled_mp3_local',
    'test': 'unlabeled_mp3_local'
}
OUTPUT_FILES = {
    'train': 'labeled_wav_local',
    'test': 'unlabeled_wav_local'
}
OUTPUT_PATH = os.path.abspath(os.path.join(cwd, '../../data/output/wavs'))
HEADER = ['title', 'audio_file', 'length', 'label']
fname_pattern = re.compile('^.*/(.*)\.mp3$')
N_CORES = 4


def transform_audio(labeled, row):
    '''Transform audio file'''
    result = {
        'title': row['title'],
        'audio_file': None,
        'length': None,
        'label': None
    }

    dataset = 'train' if labeled else 'test'
    m = fname_pattern.match(row['audio_file'])
    if not m:
        print("Could not parse the mp3 filepath")
        return result
    fname = m.group(1)
    audio_file = '%s/%s/%s' % (
        OUTPUT_PATH,
        dataset,
        '{}.wav'.format(fname))
    result['audio_file'] = audio_file
    result['length'] = row['length']
    result['label'] = row['label']
    if os.path.isfile(audio_file):
        # Already downloaded skip
        pass
    else:
        try:
            cmd = ["sox", "-V 1", "--norm", "{}".format(row['audio_file']),
                   "-c 1", "-r 8000",
                   "{}".format(audio_file)]
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print("failed to execute. Reason %s" % (e))
    return result


def process_audios(labeled=True):
    '''transform to wav file'''
    dataset = 'train' if labeled else 'test'
    with open('%s/%s.csv' % (INPUT_PATH, OUTPUT_FILES[dataset]), 'w') as fout:
        writer = CSVKitDictWriter(fout, fieldnames=HEADER)
        writer.writeheader()
        with open('%s/%s.csv' % (INPUT_PATH, INPUT_FILES[dataset]), 'r') as f:
            reader = CSVKitDictReader(f)
            r = Parallel(n_jobs=N_CORES)(delayed(transform_audio)(labeled, row)
                                         for row in reader)
            print('finished processing {}.csv'.format(INPUT_FILES[dataset]))
            writer.writerows(r)


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
