Preprocess data
===============

## Prerequisites: python3 virtualenv created and activated

More info [here](../README.md)

## Preprocessing tasks

1. Download the mp3s from amazon S3 using two index files (labeled & unlabeled)

        $ python scripts/download_audio_files.py

2. Convert to wav format in order to use python libraries like _sciPy wavfile_

        $ python scripts/convert_wav.py

3. Generate an image of the wavefiles using _librosa_ and _matplotlib_

        $ python scripts/generate_audio_image.py

4. Engineer image features:

        $ python scripts/white_proportion_average.py

5. Engineer audio features:

        $ python scripts/featurize.py

## Implementation notes

* There is a need to have Python as a framework to generate images from audio with _matplotlib_ version 1.5.0, sometimes it is not as easy when working in a virtualenv. please read [this](http://matplotlib.org/faq/virtualenv_faq.html) and choose a workaround if you find problems executing the image generation. We are using matplotlib 1.4.3 to avoid having to handle that issue

