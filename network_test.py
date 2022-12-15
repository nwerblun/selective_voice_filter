import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import wave

model = keras.models.load_model("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\model.h5")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
roots = [r"C:\Users\NWerblun\Desktop\selective_voice_filter\test_data\delibab-20071029_dump", r"C:\Users\NWerblun\Desktop\selective_voice_filter\test_data\nick_test_dump"]
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
    fft = np.fft.fft(data)
    #keep only pos half of mag. spec.
    fft = np.abs(fft).astype(np.float32)[:len(fft)//2]
    fft = fft.reshape((1,-1))
    pred = model.predict(fft, batch_size=1)
    print("Prediction for {}: {}, softmax out: {}".format(filename, pred, tf.keras.activations.sigmoid(pred)))
