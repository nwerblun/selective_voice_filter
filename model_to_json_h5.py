import os
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\model.h5")
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer="adam", metrics=["accuracy"])

with open("model_arch.json", "w") as outfile:
    arch = model.to_json()
    outfile.write(arch)
    outfile.close()

model.save_weights("model_weights.h5")
