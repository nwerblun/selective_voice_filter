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
SILENCE_DATASET_PATH = ROOT_DATASET_PATH+"\\silence_data"
ACCEPTED_SPEAKER_FOLDER_NAMES = ["nick_dump"]
VALIDATION_SPLIT = 0.2 #% of total to save for val.
SHUFFLE_SEED = 152
NOISE_SCALE_MAX = 0.25
BATCH_SIZE = 128
EPOCHS = 100
FILE_LEN = 1 #seconds
FS = 44100 #Hz

#Helper method just so I can listen to some data.
def _dump_to_file(ds):
    me_flag = 0
    not_me_flag = 0
    for tup in ds.as_numpy_iterator():
        if me_flag and not_me_flag:
            break
        if tup[1] == 1 and not me_flag:
            f = wave.open(".\\test_me.wav", "wb")
            f.setparams((1, 2, FS, len(tup[0]), "NONE", "not compressed"))
            f.writeframes(tup[0].astype(np.int16).tobytes())
            f.close()
            me_flag = 1
        elif tup[1] == 0 and not not_me_flag:
            f = wave.open(".\\test_not_me.wav", "wb")
            f.setparams((1, 2, FS, len(tup[0]), "NONE", "not compressed"))
            f.writeframes(tup[0].astype(np.int16).tobytes())
            f.close()
            not_me_flag = 1
#Helper to verify that the labels match the directory name
def _test_correct_labels(file_paths, labels, name="set"):
    errors = 0
    correct = 0
    wrong = []
    print("checking for label errors...")
    for ind, tup in enumerate(zip(file_paths, labels)):
        fpth, _ = os.path.split(tup[0])
        if "nick_dump" in fpth and tup[1] != 1:
            wrong += [tup]
            errors += 1
        elif not ("nick_dump" in fpth) and tup[0] == 1:
            wrong += [tup]
            errors += 1
        else:
            correct += 1

    print("Found {} errors and {} correct in {}.".format(errors, correct, name))
    return errors == 0 and correct == len(file_paths)

#Maybe don't use this one. Probably smarter to measure at the same audio levels
def scale_audio_volume(data, scale_min=0.25, scale_max=1.0, prob_of_scaling=0.25):
    #Pick a number from 1 - 100. Np rand is [low, high)
    np_data = data.numpy().astype(np.float32)
    do_i_scale = np.random.randint(1, 101)
    if do_i_scale <= prob_of_scaling*100:
        scale_amount = (scale_max - scale_min) * np.random.random() + scale_min
        np_data *= scale_amount
    return tf.convert_to_tensor(np_data, dtype=tf.float32)

def normalize_audio_volume(data, rms_in_dB=-10):
    np_data = data.numpy()
    rms = np.sqrt(np.mean(np_data**2))
    #Catch pure silence examples
    if rms == 0:
        return tf.convert_to_tensor(data, dtype=tf.float32)
    #assume 16 bit samples, add functionality some other time maybe
    #hardcode the max possible sample size (2^16)/2 to save some time.
    dBFS = 10*np.log10(rms/32768.0)
    gain = 10**((rms_in_dB - dBFS)/10)
    np_data *= gain
    return tf.convert_to_tensor(np_data, dtype=tf.float32)

def get_audio_from_path(file_path):
    #Since using Datasets, input will come in as a tensor object. Convert to np.
    #np converted str comes in as a bytes object, need to decode.
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
    #np str comes in as a bytes object, need to decode.
    f = wave.open(noise_paths[ind].numpy().decode('utf-8'), "rb")
    noise_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)
    prop =  np.max(audio_data.numpy()) / np.max(noise_data) # how much louder is audio
    noisy_audio = audio_data.numpy() + (scale * prop * noise_data)
    return tf.convert_to_tensor(noisy_audio, dtype=tf.float32)

def get_fft(audio):
    fft = np.fft.fft(audio.numpy())
    #keep only pos half of mag. spec.
    fft = np.abs(fft).astype(np.float32)[:len(fft)//2]
    #Reshaping to make tf happy
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
print("Loading noise directories...")
noise_paths = []
_, subdirs, _ = next(os.walk(NOISE_DATASET_PATH))
for s in subdirs:
    _, _, filenames = next(os.walk(os.path.join(NOISE_DATASET_PATH, s)))
    for f in filenames:
        if os.path.splitext(f)[1] == ".wav":
            noise_paths += [os.path.join(NOISE_DATASET_PATH, s, f)]

#Randomly set aside a few noise paths for testing and not just corrupting clips
rng = np.random.RandomState(SHUFFLE_SEED*12)
rng.shuffle(noise_paths)
noise_for_test = noise_paths[:100]
noise_paths = noise_paths[100:]
"""
Makes an assumption that the voice directory is structured as:
VOICE_ROOT/
..speaker1/
....file1.wav
....file2.wav
..speaker2/
....file1.wav
etc.

Silence directory should be
SILENCE_ROOT/
..silence1.wav
..silence2.wav
etc.
"""
print("Scanning audio files. Separating into categories and scanning for silence...")
audio_paths = []
accepted_speaker_audio_paths = []
_, _, filenames = next(os.walk(SILENCE_DATASET_PATH))
for f in filenames:
    if os.path.splitext(f)[1] == ".wav":
        audio_paths += [os.path.join(SILENCE_DATASET_PATH, f)]

print("Detected {} silence clips. Moving on to speaking clips.".format(len(audio_paths)))
_, subdirs, _ = next(os.walk(VOICE_DATASET_PATH))
for s in subdirs:
    _, _, filenames = next(os.walk(os.path.join(VOICE_DATASET_PATH, s)))
    for f in filenames:
        if os.path.splitext(f)[1] == ".wav":
            if s in ACCEPTED_SPEAKER_FOLDER_NAMES:
                accepted_speaker_audio_paths += [os.path.join(VOICE_DATASET_PATH, s, f)]
            else:
                audio_paths += [os.path.join(VOICE_DATASET_PATH, s, f)]

#Combine testing noise paths to audio clips
audio_paths = audio_paths + noise_for_test
print("Noise paths contains {} files in {} directories.".format(len(noise_paths), len(os.listdir(NOISE_DATASET_PATH))))
print("Voice paths contains {} files.".format(len(audio_paths)))
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
assert _test_correct_labels(train_audio_paths, train_labels, "pre-shuffle train set")
assert _test_correct_labels(valid_audio_paths, valid_labels, "pre-shuffle valid set")

num_train_samples = len(train_audio_paths)
num_valid_samples = len(valid_audio_paths)
print("{} training samples and {} valid samples".format(num_train_samples,  num_valid_samples))


#Use a seed to make sure they are shuffled the same and the labels still match
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(train_audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(train_labels)

rng = np.random.RandomState(SHUFFLE_SEED*2)
rng.shuffle(valid_audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED*2)
rng.shuffle(valid_labels)

assert _test_correct_labels(train_audio_paths, train_labels, "post-shuffle train set")
assert _test_correct_labels(valid_audio_paths, valid_labels, "post-shuffle valid set")

train_ds = to_ds(train_audio_paths, train_labels)
valid_ds = to_ds(valid_audio_paths, valid_labels)

# Add noise to the training set
train_ds = train_ds.map(
    lambda x, y: (tf.py_function(add_noise, [x, noise_paths, NOISE_SCALE_MAX], tf.float32), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)

#Normalize all audio clips after adding noise
train_ds = train_ds.map(
    lambda x, y: (tf.py_function(normalize_audio_volume, [x, -10.0], tf.float32), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)

train_ds = train_ds.map(
    lambda x, y: (tf.py_function(get_fft, [x], tf.float32), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

#Normalize all audio clips
valid_ds = valid_ds.map(
    lambda x, y: (tf.py_function(normalize_audio_volume, [x,-10.0], tf.float32), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

valid_ds = valid_ds.map(
    lambda x, y: (tf.py_function(get_fft, [x], tf.float32), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

"""
Very important that repeat comes before batching. These are execution graphs,
not executed at runtime, so it doesn't happen until it's needed.
If your data is something like [1,2,3...,10] and you batch by 3 then you get
[1,2,3], [4,5,6], [7,8,9], [10]
the last will give you an error when going into the network. Repeating after
just copies the [10] instead of extending it. So repeat first, then batch to get
[1,2,3], [4,5,6], [7,8,9], [10,1,2]...etc
"""
train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).repeat().batch(BATCH_SIZE)
valid_ds = valid_ds.shuffle(buffer_size=int(BATCH_SIZE*VALIDATION_SPLIT) * 8, seed=SHUFFLE_SEED).repeat().batch(int(BATCH_SIZE*VALIDATION_SPLIT))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
input_shape = (int(FS*FILE_LEN/2),1)

"""
model = keras.Sequential(
    [
        keras.Input(shape=input_shape, name="Input"),
        keras.layers.Conv1D(16, kernel_size=4, activation="relu", padding="same"),
        keras.layers.MaxPool1D(pool_size=2, strides=2),
        keras.layers.Dropout(0.15),
        keras.layers.Conv1D(32, kernel_size=4, activation="relu", padding="same"),
        keras.layers.MaxPool1D(pool_size=2, strides=2),
        keras.layers.Dropout(0.1),
        keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        keras.layers.MaxPool1D(pool_size=2, strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation=None, name="output")
    ]
)
"""
def _make_layer_helper(inp, conv_filt):
    s1 = keras.layers.Conv1D(conv_filt, kernel_size=3, padding="same")(inp)
    l1 = keras.layers.Conv1D(conv_filt, kernel_size=3, padding="same")(inp)
    l1 = keras.layers.Activation("relu")(l1)
    l1 = keras.layers.Conv1D(conv_filt, kernel_size=3, padding="same")(l1)
    l1 = keras.layers.Activation("relu")(l1)
    l1 = keras.layers.Add()([l1, s1])
    l1 = keras.layers.Activation("relu")(l1)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(l1)

#Attempt 2, non-sequential.
inp = keras.layers.Input(shape=input_shape, name="Input")
lrs = _make_layer_helper(inp, 16)
lrs = _make_layer_helper(lrs, 32)
lrs = _make_layer_helper(lrs, 64)
lrs = _make_layer_helper(lrs, 128)
lrs = keras.layers.AveragePooling1D(pool_size=4, strides=4)(lrs)
lrs = keras.layers.Dropout(0.15)(lrs)
lrs = keras.layers.Flatten()(lrs)
lrs = keras.layers.Dense(256, activation="relu")(lrs)
lrs = keras.layers.Dense(128, activation="relu")(lrs)
lrs = keras.layers.Dense(64, activation="relu")(lrs)
lrs = keras.layers.Dense(32, activation="relu")(lrs)
lrs = keras.layers.Dense(16, activation="relu")(lrs)
lrs = keras.layers.Dense(8, activation="relu")(lrs)
outs = keras.layers.Dense(1, activation=None, name="Output")(lrs)
model = keras.models.Model(inputs=inp, outputs=outs)
#Final activation is none since I'm using logits and they need to range \
#from -inf to inf

model.summary()
#Switch to logits instead of [0,1]
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer="adam", metrics=["accuracy"])
#model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model_save_filename = "model.h5"
early_cb = keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)
mid_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
)
backup_cb = keras.callbacks.BackupAndRestore(backup_dir=".\\tmp\\backup")

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    callbacks=[early_cb, mid_cb, backup_cb],
    steps_per_epoch=num_train_samples//BATCH_SIZE,
    validation_steps=num_valid_samples//int(BATCH_SIZE*VALIDATION_SPLIT)
)

print("Evaluation", model.evaluate(valid_ds, steps=num_valid_samples//int(BATCH_SIZE*VALIDATION_SPLIT)))
