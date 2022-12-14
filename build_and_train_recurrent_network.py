import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import wave
from scipy import signal
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
SHUFFLE_SEED = 6233
NOISE_SCALE_MAX = 0.25
BATCH_SIZE = 200
EPOCHS = 50
FILE_LEN = 1 #seconds
FS = 44100 #Hz
SEQUENCE_LENGTH = 5

#Helper to verify that the labels match the directory name
def _test_correct_labels(file_paths, labels, name="set"):
    #File paths is an array of SEQUENCE_LENGTH file names
    errors = 0
    correct = 0
    wrong = []

    print("checking for label errors...")
    for sequence_names, label in zip(file_paths, labels):
        for fname in sequence_names:
            fpth, _ = os.path.split(fname)
            if "nick_dump" in fpth and label != 1:
                wrong += [(fname, label)]
                errors += 1
            elif not ("nick_dump" in fpth) and label == 1:
                wrong += [(fname, label)]
                errors += 1
            else:
                correct += 1

    print("Found {} errors and {} correct in {}.".format(errors, correct, name))
    if errors > 0:
        print("Found errors in these files:")
        for i in wrong:
            print(i)
    return errors == 0 and correct == file_paths.shape[0]*file_paths.shape[1]

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
    normed = []
    for d in data:
        np_data = d.numpy()
        rms = np.sqrt(np.mean(np_data**2))
        #Catch pure silence examples
        if rms == 0:
            normed += [np_data]
            continue
        #assume 16 bit samples, add functionality some other time maybe
        #hardcode the max possible sample size (2^16)/2 to save some time.
        dBFS = 10*np.log10(rms/32768.0)
        gain = 10**((rms_in_dB - dBFS)/10)
        np_data *= gain
        normed += [np_data]
    return tf.convert_to_tensor(np.array(normed), dtype=tf.float32)

def get_audio_from_path(file_paths):
    #Since using Datasets, input will come in as a tensor object. Convert to np.
    #np converted str comes in as a bytes object, need to decode.
    sequence_paths = file_paths.numpy()
    sequence_audios = []
    for fname in sequence_paths:
        f = wave.open(fname.decode('utf-8'), "rb")
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        f.close()
        sequence_audios += [data.astype(np.float32)]
    return tf.convert_to_tensor(np.array(sequence_audios), dtype=tf.float32)

def to_ds(paths, labels):
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    audio_ds = paths_ds.map(lambda x: tf.py_function(get_audio_from_path, [x], tf.float32))
    return tf.data.Dataset.zip((audio_ds, labels_ds))

def add_noise(audio_data, noise_paths, scale_max=0.5):
    noisy_audio = []
    for a in audio_data:
        #choose a random noise
        ind = np.random.randint(0, len(noise_paths))

        #Trying scaling all noise by the same amount. Let's see how that goes.
        #scale = np.random.uniform(0.01, scale_max)

        #Since using Datasets, input will come in as a tensor object. Convert to np.
        #np str comes in as a bytes object, need to decode.
        f = wave.open(noise_paths[ind].numpy().decode('utf-8'), "rb")
        noise_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)
        prop =  np.max(np.abs(a.numpy())) / np.max(np.abs(noise_data))
        #noisy_audio = audio_data.numpy() + (scale * prop * noise_data)
        noisy_audio += [a.numpy() + (scale_max * prop * noise_data)]
    return tf.convert_to_tensor(np.array(noisy_audio))

def get_fft(audio):
    fft = np.fft.fft(audio.numpy())
    #keep only pos half of mag. spec.
    fft = np.abs(fft).astype(np.float32)[:len(fft)//2]
    #Reshaping to make tf happy
    return tf.convert_to_tensor(fft.reshape((fft.shape[0],1)))

def get_spectrogram(audio):
    spectrograms = []
    for a in audio:
        _, _, Sxx = signal.spectrogram(a.numpy(), fs=FS, nperseg=512, mode="magnitude")
        #Add tiny value to avoid 0
        scaled = 10*np.log10(Sxx+1e-9)
        #explicitly show channels
        spectrograms += [scaled]
    new_shape = (len(audio), spectrograms[0].shape[0], spectrograms[0].shape[1], 1)
    return tf.convert_to_tensor(np.array(spectrograms).reshape(new_shape), dtype=tf.float32)

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
"""
print("Scanning audio files. Separating into categories and scanning for silence...")
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

#Combine testing noise paths to audio clips
audio_paths = audio_paths + noise_for_test
print("Noise paths contains {} files in {} directories.".format(len(noise_paths), len(os.listdir(NOISE_DATASET_PATH))))
print("Voice paths contains {} files.".format(len(audio_paths)))
print("Found {} clips from the accepted speaker.".format(len(accepted_speaker_audio_paths)))

#Drop samples if we can't evenly do it
accepted_speaker_audio_paths = accepted_speaker_audio_paths[:SEQUENCE_LENGTH*(len(accepted_speaker_audio_paths)//SEQUENCE_LENGTH)]
#Reshape into mini sequences
accepted_speaker_audio_paths = np.array(accepted_speaker_audio_paths).reshape((len(accepted_speaker_audio_paths)//SEQUENCE_LENGTH, SEQUENCE_LENGTH))

#Drop samples if we can't evenly do it
audio_paths = audio_paths[:SEQUENCE_LENGTH*(len(audio_paths)//SEQUENCE_LENGTH)]
#Reshape into mini sequences
audio_paths = np.array(audio_paths).reshape((len(audio_paths)//SEQUENCE_LENGTH, SEQUENCE_LENGTH))

#Split into two sets
num_split = int(VALIDATION_SPLIT * len(audio_paths))
num_split2 = int(VALIDATION_SPLIT * len(accepted_speaker_audio_paths))

train_audio_paths = np.vstack((audio_paths[:-num_split, :], accepted_speaker_audio_paths[:-num_split2, :]))
valid_audio_paths = np.vstack((audio_paths[-num_split:, :], accepted_speaker_audio_paths[-num_split2:, :]))
#Only two classes. 0 for not me, 1 for me
train_labels = [0]*len(audio_paths[:-num_split, :]) + [1]*len(accepted_speaker_audio_paths[:-num_split2 , :])
valid_labels = [0]*len(audio_paths[-num_split:, :]) + [1]*len(accepted_speaker_audio_paths[-num_split2:, :])
assert len(train_audio_paths) == len(train_labels)
assert len(valid_audio_paths) == len(valid_labels)
assert _test_correct_labels(train_audio_paths, train_labels, "pre-shuffle train set")
assert _test_correct_labels(valid_audio_paths, valid_labels, "pre-shuffle valid set")

num_train_samples = len(train_audio_paths)
num_valid_samples = len(valid_audio_paths)
print("{} training samples and {} valid samples".format(num_train_samples,  num_valid_samples))

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
    lambda x, y: (tf.py_function(get_spectrogram, [x], tf.float32), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
#Normalize all audio clips
valid_ds = valid_ds.map(
    lambda x, y: (tf.py_function(normalize_audio_volume, [x,-10.0], tf.float32), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

valid_ds = valid_ds.map(
    lambda x, y: (tf.py_function(get_spectrogram, [x], tf.float32), y),
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
train_ds = train_ds.repeat().batch(BATCH_SIZE)
valid_ds = valid_ds.repeat().batch(int(BATCH_SIZE*VALIDATION_SPLIT))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
input_shape_cnn = (512//2+1, int(FS/(512-512//8)),1)
input_shape_rnn = (SEQUENCE_LENGTH, 512//2+1, int(FS/(512-512//8)),1)

#Attempt 4, Recurrent.
inp_cnn = keras.layers.Input(shape=input_shape_cnn, name="CNN_Input")
lrs = keras.layers.Conv2D(16, kernel_size=(3,3), strides=3, activation="relu", padding="same")(inp_cnn)
lrs = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(lrs)

lrs = keras.layers.Conv2D(32, kernel_size=(5,5), strides=5, activation="relu", padding="same")(lrs)
lrs = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(lrs)

lrs = keras.layers.Conv2D(64, kernel_size=(5,5), strides=5, activation="relu", padding="same")(lrs)
lrs = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(lrs)

lrs = keras.layers.Dropout(0.2)(lrs)
out_cnn = keras.layers.Flatten(name="Out_CNN")(lrs)
cnn = keras.models.Model(inputs=inp_cnn, outputs=out_cnn)
cnn.summary()

#Recurrent
inp_rnn = keras.layers.Input(shape=input_shape_rnn, name="RNN_Input")
rnn_lrs = keras.layers.TimeDistributed(cnn)(inp_rnn)
rnn_lrs = keras.layers.GRU(128, return_sequences=True)(rnn_lrs)
rnn_lrs = keras.layers.GRU(256, return_sequences=True)(rnn_lrs)
rnn_lrs = keras.layers.Flatten()(rnn_lrs)
rnn_lrs = keras.layers.Dense(128, activation="relu")(rnn_lrs)
out_rnn = keras.layers.Dense(1, activation=None, name="Out_RNN")(rnn_lrs)
model = keras.models.Model(inputs=inp_rnn, outputs=out_rnn)
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
