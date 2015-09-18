ML classification - Audio files
===============================

## Introduction

The goal of this analysis is to try to use ML to automatically classify audios with telephone conversations leaked.

In particular we wanted to discard answering machines and non answered calls so that we could reduce the amount of files that a human needs to audit searching for interesting information.

In order to extract image related features we are using [OpenCV 3.0][cv] in order to
install it from source linked with python3 follow [these][cvpy] instructions

In order to extract audio related features we are usign [librosa][librosa]

## Feature engineering 

Since our goal was to discard audios that went to answering machines or where not responded we started to think about the characteristics of those audios.

Percent of silence seemed to be one of the first things that came to mind, also if we detected the ringtones we could use the length of the audio from the last ring as a feature for our classification.

### Audio features 
- Ring detection
- Percentage of silence
- Length of the chunk between the last ring and the end of the file
- Number of rings

### Image features
* white proportion
    * We have computed the percentage of white available in each image waveform, the greater the value the stronger the possibility of the audio being a non interesting file... or it least that is our hypothesis

[cv]: http://opencv.org/
[cvpy]: http://www.pyimagesearch.com/2015/06/29/install-opencv-3-0-and-python-3-4-on-osx/
[librosa]: https://github.com/bmcfee/librosa

## Repo structure

The repo is structured in three main folders

### Preprocess (Python scripts)

Scripts used to download the audios and do the feature extraction

### Analysis (Jupyter notebooks)

Intermediate Analysis to help us understand our dataset characteristics and the performance of our selected features

### Classification (Jupyter notebooks)

Final classification process, also a validation notebook to manually check the overall performance of the Machine Learning process.

## Authors
* Juan Elosua
* Francis Tzeng