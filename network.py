import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import wave
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
FILE_LEN = 0.5 #seconds
FS = 44100 #Hz

def get_audio_from_path(file_path):
    #Since using Datasets, input will come in as a tensor object. Convert to np.
    #str comes in as a bytes object, need to decode.
    f = wave.open(file_path.numpy().decode('utf-8'), "rb")
    data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    f.close()
    return tf.convert_to_tensor(data.astype(np.float32), dtype=tf.float32)

def to_ds(paths, labels):
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    audio_ds = paths_ds.map(lambda x: tf.py_function(get_audio_from_path, [x], tf.float32))
    return tf.data.Dataset.zip((audio_ds, labels_ds))

def add_noise(audio_data, noise_paths, scale_max=0.5):
    #choose a random noise
    ind = np.random.randint(0, len(noise_paths))
    scale = np.random.uniform(0, scale_max)
    #Since using Datasets, input will come in as a tensor object. Convert to np.
    #str comes in as a bytes object, need to decode.
    f = wave.open(noise_paths[ind].numpy().decode('utf-8'), "rb")
    noise_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)
    prop =  np.max(audio_data.numpy()) / np.max(noise_data) # how much louder is audio
    noisy_audio = audio_data.numpy() + (scale * prop * noise_data)
    return tf.convert_to_tensor(noisy_audio, dtype=tf.float32)

def get_fft(audio):
    fft = np.fft.fft(audio.numpy())
    #keep only pos half of mag. spec.
    fft = np.abs(fft).astype(np.float32)[:len(fft)//2]
    return tf.convert_to_tensor(fft.reshape((fft.shape[0],1)))

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

num_train_samples = len(train_audio_paths)
num_valid_samples = len(valid_audio_paths)
print("{} training samples and {} valid samples".format(num_train_samples,  num_valid_samples))

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

# Add noise to the training set
#TODO: Remove the comments that block out the autotune things. I cancelled them so I could debug
train_ds = train_ds.map(
    lambda x, y: (tf.py_function(add_noise, [x, noise_paths, NOISE_SCALE_MAX], tf.float32), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)

train_ds = train_ds.map(
    lambda x, y: (tf.py_function(get_fft, [x], tf.float32), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

valid_ds = valid_ds.map(
    lambda x, y: (tf.py_function(get_fft, [x], tf.float32), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 16, seed=SHUFFLE_SEED).repeat().batch(BATCH_SIZE)
valid_ds = valid_ds.shuffle(buffer_size=int(BATCH_SIZE*VALIDATION_SPLIT) * 16, seed=SHUFFLE_SEED).repeat().batch(int(BATCH_SIZE*VALIDATION_SPLIT))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
input_shape = (int(FS*FILE_LEN/2),1)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape, name="Input"),
        keras.layers.Conv1D(16, kernel_size=3, activation="relu", padding="same"),
        keras.layers.MaxPool1D(pool_size=2, strides=2),
        keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        keras.layers.MaxPool1D(pool_size=2, strides=2),
        keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        keras.layers.MaxPool1D(pool_size=2, strides=2),
        keras.layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        keras.layers.MaxPool1D(pool_size=2, strides=2),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="softmax", name="output")
    ]
)

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model_save_filename = "model.h5"
early_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
mid_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
)


history = model.fit(
    train_ds,
    epochs=EPOCHS, #Idk I just want it to stop crashing from running out of data
    validation_data=valid_ds,
    callbacks=[early_cb, mid_cb],
    steps_per_epoch=num_train_samples//BATCH_SIZE,
    validation_steps=num_valid_samples//int(BATCH_SIZE*VALIDATION_SPLIT)
)

print("Evaluation", model.evaluate(valid_ds))
