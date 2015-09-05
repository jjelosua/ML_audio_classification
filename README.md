Nisman audio files - ML classification
======================================

## Intro

The goal of this analysis is to try to automatically classify the hearing audios leaked corresponding to
the nisman investigation.

In particular we wanted to discard answering machines and non answered calls so that we could reduce the amount of files that a human needs to audit searching for interesting information.

In order to extract image related features we are using [OpenCV 3.0][cv] in order to
install it from source linked with python3 follow [these][cvpy] instructions

## Audio features
-Ring detection
-Percentage of silence
-Length of the chunk between the last ring and the end of the file
-Number of rings

## Image features
* white proportion
    * We have computed the percentage of white availabel in each image waveform, the greater the value
    the stronger the possibility of the audio being a non interesting file....or it least that is our
    hypothesis

[cv]: http://opencv.org/
[cvpy]: http://www.pyimagesearch.com/2015/06/29/install-opencv-3-0-and-python-3-4-on-osx/

## TODO
Document the process better

## Authors
* Juan Elosua
* Francis Tzeng
