import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow import keras
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
ACCEPTED_SPEAKER_FOLDER_NAMES = ["nick_dump"]
VALIDATION_SPLIT = 0.2 #% of total to save for val.
SHUFFLE_SEED = 152
NOISE_SCALE_MAX = 0.7
BATCH_SIZE = 128
EPOCHS = 150

def get_audio_from_path(path):
    f = wave.open(path, "rb")
    data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    f.close()
    return data.astype(np.float32)

def to_ds(paths, labels):
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    audio_ds = paths_ds.map(lambda x: get_audio_from_path(x))
    return tf.data.Dataset.zip((audio_ds, labels_ds))

def add_noise(audio_data, noise_paths, scale_max=0.5):
    #choose a random noise
    ind = np.random.randint(0, len(noise_paths))
    scale = np.random.uniform(0, scale_max)
    f = wave.open(noise_paths[ind], "rb")
    noise_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16),astype(np.float32)
    prop =  audio_data / noise_data # how much louder is noise than audio
    noisy_audio = audio_data + (scale * prop * noise_data)
    return noisy_audio

def get_fft(audio):
    fft = np.fft.fft(audio)
    #Get magnitude spectrum, return only first half. Ignore neg. freqs.
    return np.abs(fft)[:len(fft)//2]

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
noise_paths = []
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
accepted_speaker_audio_paths = []

_, subdirs, _ = next(os.walk(VOICE_DATASET_PATH))
for s in subdirs:
    _, _, filenames = next(os.walk(os.path.join(VOICE_DATASET_PATH, s)))
    for f in filenames:
        if os.path.splitext(f)[1] == ".wav":
            if s in ACCEPTED_SPEAKER_FOLDER_NAMES:
                accepted_speaker_audio_paths += [os.path.join(VOICE_DATASET_PATH, s, f)]
            else:
                audio_paths += [os.path.join(VOICE_DATASET_PATH, s, f)]

print("Noise paths contains {} files in {} directories.".format(len(noise_paths), len(os.listdir(NOISE_DATASET_PATH))))
print("Voice paths contains {} files belonging to {} speakers.".format(len(audio_paths), len(os.listdir(VOICE_DATASET_PATH))))
print("Found {} clips from the accepted speaker.".format(len(accepted_speaker_audio_paths)))

#Split into two sets
num_split = int(VALIDATION_SPLIT * len(audio_paths))
num_split2 = int(VALIDATION_SPLIT * len(accepted_speaker_audio_paths))

train_audio_paths = audio_paths[:-num_split] + accepted_speaker_audio_paths[:-num_split2]
valid_audio_paths = audio_paths[-num_split:] + accepted_speaker_audio_paths[-num_split2:]
#Only two classes. 0 for not me, 1 for me
train_labels = [0]*len(audio_paths[:-num_split]) + [1]*len(accepted_speaker_audio_paths[:-num_split2])
valid_labels = [0]*len(audio_paths[-num_split:]) + [1]*len(accepted_speaker_audio_paths[-num_split2:])
assert len(train_audio_paths) == len(train_labels)
assert len(valid_audio_paths) == len(valid_labels)

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(train_audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(train_labels)

rng = np.random.RandomState(SHUFFLE_SEED*2)
rng.shuffle(valid_audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED*2)
rng.shuffle(valid_labels)

train_ds = to_ds(train_audio_paths, train_labels)
valid_ds = to_ds(valid_audio_paths, valid_labels)

train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
valid_ds = valid_ds.shuffle(buffer_size=int(BATCH_SIZE * VALIDATION_SPLIT * 4), seed=SHUFFLE_SEED).batch(int(BATCH_SIZE * VALIDATION_SPLIT))

# Add noise to the training set
train_ds = train_ds.map(
    lambda x, y: (add_noise(x, noise_paths, scale_max=NOISE_SCALE_MAX), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_ds.map(
    lambda x, y: (get_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

valid_ds = valid_ds.map(
    lambda x, y: (get_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

input_shape = (44100//2,)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
