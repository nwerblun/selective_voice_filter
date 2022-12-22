import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import wave

from_h5 = True
from_json = not from_h5

if from_h5:
    model = keras.models.load_model("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\model.h5")
if from_json:
    f = open("model_arch.json", "r")
    model_text = f.read()
    f.close()
    model = keras.models.model_from_json(model_text)
    model.load_weights("model_weights.h5")

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
roots = [
        r"C:\Users\NWerblun\Desktop\selective_voice_filter\test_data\delibab-20071029_dump", \
        r"C:\Users\NWerblun\Desktop\selective_voice_filter\test_data\calamity-20071011-poe_dump", \
        r"C:\Users\NWerblun\Desktop\selective_voice_filter\test_data\Jens_Stoltenberg", \
        r"C:\Users\NWerblun\Desktop\selective_voice_filter\test_data\noise", \
        r"C:\Users\NWerblun\Desktop\selective_voice_filter\test_data\silence", \
        r"C:\Users\NWerblun\Desktop\selective_voice_filter\test_data\nick_test_dump"
]
all_files = []
for dir in roots:
    _,_,filenames = next(os.walk(dir))
    for f in filenames:
        all_files.append(os.path.join(dir, f))

for filename in all_files:
    f = wave.open(filename, "rb")
    data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    f.close()
    data = data.astype(np.float32)
    rms_audio = np.sqrt(np.mean(data**2))
    if rms_audio != 0:
        dBFS = 10*np.log10(rms_audio/(2**15))
        scaled_speak = data * (10**((-10 - dBFS)/10))
    else:
        scaled_speak = data

    fft = np.fft.fft(scaled_speak)
    #keep only pos half of mag. spec.
    fft = np.abs(fft).astype(np.float32)[:len(fft)//2]
    fft = fft.reshape((1,-1))
    pred = model.predict(fft, batch_size=1)
    print("Prediction for {}: {}, sigmoid out: {}".format(filename, pred, tf.keras.activations.sigmoid(pred)))
