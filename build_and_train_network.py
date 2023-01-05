import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
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
ACCEPTED_SPEAKER_FOLDER_NAMES = ["nick_dump"]
VALIDATION_SPLIT = 0.2 #% of total to save for val.
SHUFFLE_SEED = 6233
BATCH_SIZE = 200
EPOCHS = 50
FILE_LEN = 1 #seconds
FS = 16000 #Hz
SPEC_WINDOW_LENGTH = 256
SPEC_OVERLAP = 128
NFFT = 256

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

def get_spec_from_path(file_path):
    #Since using Datasets, input will come in as a tensor object. Convert to np.
    #np converted str comes in as a bytes object, need to decode.
    im = Image.open(file_path.numpy().decode('utf-8'))
    rgb = np.array(list(im.getdata()))
    time_chunks = int((FS/SPEC_WINDOW_LENGTH) + ((FS-SPEC_OVERLAP)/SPEC_WINDOW_LENGTH))
    rgb = rgb.reshape((
        NFFT//2+1,
        time_chunks,
        3
    ))
    return tf.convert_to_tensor(rgb, dtype=tf.uint8)

def to_ds(paths, labels):
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    spec_ds = paths_ds.map(lambda x: tf.py_function(get_spec_from_path, [x], tf.uint8))
    return tf.data.Dataset.zip((spec_ds, labels_ds))

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
        if os.path.splitext(f)[1] == ".png":
            if s in ACCEPTED_SPEAKER_FOLDER_NAMES:
                accepted_speaker_audio_paths += [os.path.join(VOICE_DATASET_PATH, s, f)]
            else:
                audio_paths += [os.path.join(VOICE_DATASET_PATH, s, f)]

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


"""
Very important that repeat comes before batching. These are execution graphs,
not executed at runtime, so it doesn't happen until it's needed.
If your data is something like [1,2,3...,10] and you batch by 3 then you get
[1,2,3], [4,5,6], [7,8,9], [10]
the last will give you an error when going into the network. Repeating after
just copies the [10] instead of extending it. So repeat first, then batch to get
[1,2,3], [4,5,6], [7,8,9], [10,1,2]...etc
"""
train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 6, seed=SHUFFLE_SEED).repeat().batch(BATCH_SIZE)
valid_ds = valid_ds.shuffle(buffer_size=int(BATCH_SIZE*VALIDATION_SPLIT) * 6, seed=SHUFFLE_SEED).repeat().batch(int(BATCH_SIZE*VALIDATION_SPLIT))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
time_chunks = int((FS/SPEC_WINDOW_LENGTH) + ((FS-SPEC_OVERLAP)/SPEC_WINDOW_LENGTH))
input_shape = (
        NFFT//2+1,
        time_chunks,
        3
    )

#Attempt 3, non-sequential but way smaller.
inp = keras.layers.Input(shape=input_shape, name="Input")

lrs = keras.layers.Conv2D(16, kernel_size=(5,5), strides=1, activation="relu", padding="same")(inp)
lrs = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(lrs)

lrs = keras.layers.Conv2D(32, kernel_size=(3,3), strides=1, activation="relu", padding="same")(lrs)
lrs = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(lrs)

lrs = keras.layers.Conv2D(64, kernel_size=(3,3), strides=1, activation="relu", padding="same")(lrs)
lrs = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(lrs)

lrs = keras.layers.Conv2D(64, kernel_size=(3,3), strides=1, activation="relu", padding="same")(lrs)
lrs = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(lrs)

lrs = keras.layers.Conv2D(64, kernel_size=(3,3), strides=1, activation="relu", padding="same")(lrs)
lrs = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(lrs)

lrs = keras.layers.Dropout(0.2)(lrs)
lrs = keras.layers.Flatten()(lrs)

lrs = keras.layers.Dense(4096, activation="relu")(lrs)
lrs = keras.layers.Dense(2048, activation="relu")(lrs)
lrs = keras.layers.Dense(256, activation="relu")(lrs)
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
