# coding: utf-8
import requests
import os
from csvkit.py3 import CSVKitDictReader, CSVKitDictWriter
import shutil
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
OUTPUT_AUDIO_PATH = os.path.abspath(os.path.join(
    cwd,
    '../../data/output/mp3s'))
INPUT_FILES = {
    'train': 'labeled_audios',
    'test': 'unlabeled_audios'
}
OUTPUT_FILES = {
    'train': 'labeled_mp3_local',
    'test': 'unlabeled_mp3_local'
}
HEADER = ['audio_file', 'length', 'label']
N_CORES = 7


def download_audio(labeled, row):
    '''Download an audio file'''
    result = {}
    dataset = 'train' if labeled else 'test'
    audio_file = '%s/%s/%s' % (
        OUTPUT_AUDIO_PATH,
        dataset,
        row['Document Title'])
    result['audio_file'] = audio_file
    result['length'] = row['length']
    if labeled:
        answering = row['answer_El audio parece estar relacionado con:']
        result['label'] = 1 if answering.startswith('Descartar:') else 0
    else:
        result['label'] = None
    if os.path.isfile(audio_file):
        # Already downloaded skip
        pass
    else:
        # Download audio
        try:
            response = requests.get(row['Document Url'], stream=True)
            with open(audio_file, 'wb') as of:
                shutil.copyfileobj(response.raw, of)
            del response
        except Exception:
            print('Error while downloading {}...skipping'
                  .format(row['Document Url']))
    return result


def process_audios(labeled=True):
    '''Download audio data'''
    dataset = 'train' if labeled else 'test'
    with open('%s/%s.csv' % (INPUT_PATH, OUTPUT_FILES[dataset]), 'w') as fout:
        writer = CSVKitDictWriter(fout, fieldnames=HEADER)
        writer.writeheader()
        with open('%s/%s.csv' % (INPUT_PATH, INPUT_FILES[dataset]), 'r') as f:
            reader = CSVKitDictReader(f)
            r = Parallel(n_jobs=N_CORES)(delayed(download_audio)(labeled, row)
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
        '%s/train' % OUTPUT_AUDIO_PATH,
        # Test set audio files
        '%s/test' % OUTPUT_AUDIO_PATH]
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
