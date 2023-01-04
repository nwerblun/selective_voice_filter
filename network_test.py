import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import wave
from scipy import signal

os.system("color") #needed to show colored text
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

from_h5 = True
latest_arch = 4
latest_weights = 5

print("Loading model...")
if from_h5:
    model = keras.models.load_model("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\model.h5")
else:
    f = open("old_models\\model_arch_"+str(latest_arch)+".json", "r")
    model_text = f.read()
    f.close()
    model = keras.models.model_from_json(model_text)
    model.load_weights("old_models\\model_weights_"+str(latest_weights)+".h5")

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
root = r"C:\Users\NWerblun\Desktop\selective_voice_filter\test_data"
all_files = []

os.system("cls")
print("Beginning testing...")
_,s,_ = next(os.walk(root))
for sub in s:
    _,_,filenames = next(os.walk(os.path.join(root,sub)))
    for f in filenames:
        if sub == "nick_test_dump":
            all_files.append((os.path.join(root,sub,f), 1))
        else:
            all_files.append((os.path.join(root,sub,f), 0))

fails = 0
successes = 0
failed_files = []
for filename, label in all_files:
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

    """
    fft = np.fft.fft(scaled_speak)
    #keep only pos half of mag. spec.
    fft = np.abs(fft).astype(np.float32)[:len(fft)//2]
    fft = fft.reshape((1,-1))
    """

    _, _, Sxx = signal.spectrogram(scaled_speak, fs=44100, nperseg=512, mode="magnitude")
    #Add tiny value to avoid 0
    scaled = 10*np.log10(Sxx+1e-9)
    #explicitly state 1 batch because predict is stupid
    new_shape = (1, scaled.shape[0], scaled.shape[1])
    pred = model.predict(scaled.reshape(new_shape), batch_size=1)

    strg = "Prediction for {}: {}, sigmoid out: {}".format(filename, pred, tf.keras.activations.sigmoid(pred))
    spaces = " "*(175 - len(strg))
    strg += spaces
    if label and pred > 0:
        successes += 1
        strg += "\t" + bcolors.OKGREEN + "PASS" + bcolors.ENDC
    elif label and pred <= 0:
        fails += 1
        failed_files += [(filename, pred, label)]
        strg += "\t" + bcolors.FAIL + "FAIL" + bcolors.ENDC
    elif not label and pred > 0:
        fails += 1
        failed_files += [(filename, pred, label)]
        strg += "\t" + bcolors.FAIL + "FAIL" + bcolors.ENDC
    else:
        successes += 1
        strg += "\t" + bcolors.OKGREEN + "PASS" + bcolors.ENDC
    print(strg)

print("Failed files:")
for fname, pred, label in failed_files:
    strg = ""
    strg += fname
    strg += " "*(130 - len(strg))
    strg += "prediction: " + str(pred)
    strg += " "*(140 - len(strg))
    strg += "\tsigmoid prediction: {}".format(tf.keras.activations.sigmoid(pred))
    strg += " "*(170 - len(strg))
    strg += "correct: " + str(label)
    print(strg)
print("Total fails: {}\nTotal successes: {}\nPercentage: {}".format(fails, successes, successes/len(all_files)))
