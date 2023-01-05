import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import wave
from scipy import signal
from PIL import Image

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
SPEC_WINDOW_LENGTH = 256
FS = 16000
SPEC_OVERLAP = 128
NFFT = 256

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
    im = Image.open(filename)
    rgb = np.array(list(im.convert("RGB").getdata()))
    time_chunks = int((FS/SPEC_WINDOW_LENGTH) + ((FS-SPEC_OVERLAP)/SPEC_WINDOW_LENGTH))
    rgb = rgb.reshape((
        NFFT//2+1,
        time_chunks,
        3
    ))

    new_shape = (1, rgb.shape[0], rgb.shape[1], rgb.shape[2])
    pred = model.predict(rgb.reshape(new_shape), batch_size=1)

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
