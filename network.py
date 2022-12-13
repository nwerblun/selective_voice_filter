import numpy as np
import os
import shutil
from tensorflow import keras
from functoolks import reduce
"""
prepare data into
X =
np.array([
    [FFT of clip 1],
    [FFT of clip 2],
    ...
    [FFT of clip N]
])

Y =
np.array([
    [1 if me, 0 if not],
    [1 if me, 0 if not],
    ...
    [1 if me, 0 if not]
])

Input shape = ()
"""

ROOT_DATASET_PATH = "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data"
VOICE_DATASET_PATH = ROOT_DATASET_PATH+"\\voice_data"
NOISE_DATASET_PATH = ROOT_DATASET_PATH+"\\noise_data"
ACCEPTED_SPEAKER_FOLDER_NAME = "nick"

noise_paths = []
"""
Makes an assumption that the noise directory is structured as:
NOISE_ROOT/
..folder 1/
....noise1.wav
....noise2.wav
..folder2/
....noise1.wav
etc.
"""
_, subdirs, _ = next(os.walk(NOISE_DATASET_PATH))
for s in subdirs:
    _, _, filenames = next(os.walk(os.path.join(NOISE_DATASET_PATH, s)))
    for f in filenames:
        if os.path.splitext(f)[1] == ".wav":
            noise_paths += [os.path.join(NOISE_DATASET_PATH, s, f)]

"""
Makes an assumption that the voice directory is structured as:
VOICE_ROOT/
..speaker1/
....file1.wav
....file2.wav
..speaker2/
....file1.wav
etc.
"""
audio_paths = []
labels = []

_, subdirs, _ = next(os.walk(VOICE_DATASET_PATH))
for s in subdirs:
    _, _, filenames = next(os.walk(os.path.join(VOICE_DATASET_PATH, s)))
    for f in filenames:
        if os.path.splitext(f)[1] == ".wav":
            audio_paths += [os.path.join(VOICE_DATASET_PATH, s, f)]
            labels += [1 if s == ACCEPTED_SPEAKER_FOLDER_NAME else 0]

num_accepted_speaker_clips = reduce(lambda x, y: x+y, labels)
print("Noise paths contains {} files in {} directories.".format(len(noise_paths), len(os.listdir(NOISE_DATASET_PATH))))
print("Voice paths contains {} files belonging to {} speakers.".format(len(audio_paths), len(os.listdir(VOICE_DATASET_PATH))))
print("Found {} clips from the accepted speaker.".format(num_accepted_speaker_clips))
