# coding: utf-8
import os
from csvkit.py3 import CSVKitDictReader, CSVKitDictWriter
import numpy as np
import matplotlib.pyplot as plt
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
    'train': 'labeled_wav_local',
    'test': 'unlabeled_wav_local'
}
OUTPUT_FILES = {
    'train': 'labeled_wav_image_local',
    'test': 'unlabeled_wav_image_local'
}
OUTPUT_PATH = os.path.abspath(os.path.join(cwd, '../../data/output/images'))
HEADER = ['title', 'audio_file', 'image_file', 'length', 'label']
PLOT_DURATION = 60
SAMPLE_RATE = 8000
N_CORES = 7


def gen_picture(audio_path, image_path, duration):
        y, sr = librosa.load(audio_path, duration=duration, sr=None)

        # Fix the duration in order to generate a fair feature
        # y = librosa.util.fix_length(y, duration * SAMPLE_RATE)

        # Generate subplot
        fig, ax = plt.subplots(1)

        # Give some margin to acommodate the plot
        ax.set_ylim([-1.1, 1.1])

        # Config the specs of the image
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

        ax.set_ylabel('Amplitud')
        ax.set_xlabel('Segundos')
        ax.yaxis.label.set_color('black')
        ax.xaxis.label.set_color('black')

        # Plot the audio image
        plt.plot(np.linspace(0.0, y.size/sr, y.size), y, color='#5f8724')
        try:
            fig.savefig(image_path,
                        figsize=(8, 6),
                        facecolor='white',
                        transparent=True,
                        dpi=100)
        finally:
            plt.clf()
            plt.close()


def make_plot(labeled, row):
    '''make a plot from the audio file'''
    dataset = 'train' if labeled else 'test'
    result = {
        'title': row['title'],
        'audio_file': None,
        'image_file': None,
        'length': None,
        'label': None
    }

    audio_path = row['audio_file']
    fname = audio_path.split('/')[-1]
    fname = fname.replace('.wav', '.png')
    image_path = '{}/{}/{}'.format(OUTPUT_PATH, dataset, fname)
    result['audio_file'] = row['audio_file']
    result['image_file'] = image_path
    result['length'] = row['length']
    result['label'] = row['label']
    try:
        if not os.path.exists(image_path):
            gen_picture(audio_path, image_path, PLOT_DURATION)
    except Exception as e:
        print("found error in %s at %s. Reason %s" % (
            image_path,
            PLOT_DURATION,
            e))
    return result


def process_audios(labeled=True):
    '''extract the image of the audio file'''
    dataset = 'train' if labeled else 'test'
    with open('%s/%s.csv' % (INPUT_PATH, OUTPUT_FILES[dataset]), 'w') as fout:
        writer = CSVKitDictWriter(fout, fieldnames=HEADER)
        writer.writeheader()
        with open('%s/%s.csv' % (INPUT_PATH, INPUT_FILES[dataset]), 'r') as f:
            reader = CSVKitDictReader(f)
            r = Parallel(n_jobs=N_CORES)(delayed(make_plot)(labeled, row)
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
