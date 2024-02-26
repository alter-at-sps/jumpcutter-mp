# Jumpcutter-mp
Based on Gegell's [fork](https://github.com/Gegell/jumpcutter) of carykh's [jumpcutter](https://github.com/carykh/jumpcutter). It speeds up or slows down silent and louder parts of a video at different rates.

This fork is just a few optimalizations added to the already quite fast Gerell's fork. The changes make it mainly much more performant on very long videos. (can process 3 hrs worth of footage and audio in ~7 mins on a 12 threaded system with only ~3GB of ram)

Changes and additions:
- Descresed Memory Usage by using an Shared Audio Buffers
- Parallelized phasevocoding and fading
- Optional Voice Activity Detector (WebRTC) support for removing non-voice audio (must enable in flags)
- Optional [librosa](https://librosa.org/) based audio preprocessing and filtering used for thresholding (eg. helpful for removing non-speech audio)
  - built-in vocal filtering
  - Preprocessed audio caching (must enable in flags but heavily recommended)

May add in the future:
- more preprocessing options for librosa

## Installing dependencies
To install the python libaries this script depends on, simply run `pip install requirements.txt`.

As it also relies heavily on ffmpeg, please make sure that it is installed and accesible over the commandline. 
See [FFmpeg Website](ffmpeg.org) for information on how to install it.

## Running it
To run it call `python jumpcutter.py -h` to get a help for the settings that you can pass to it.

To simply use default settings use `python jumpcutter.py -i INPUT_FILE`.
