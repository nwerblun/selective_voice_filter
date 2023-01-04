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
SEQUENCE_LENGTH = 5

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

not_me_files = []
me_files = []

os.system("cls")
print("Beginning testing...")
_,s,_ = next(os.walk(root))
for sub in s:
    _,_,filenames = next(os.walk(os.path.join(root,sub)))
    for f in filenames:
        if sub == "nick_test_dump":
            me_files.append(os.path.join(root,sub,f))
        else:
            not_me_files.append(os.path.join(root,sub,f))

fails = 0
successes = 0
failed_files = []
#drop some files
not_me_files = not_me_files[:SEQUENCE_LENGTH*(len(not_me_files)//SEQUENCE_LENGTH)]
not_me_files = np.array(not_me_files).reshape((len(not_me_files)//SEQUENCE_LENGTH, SEQUENCE_LENGTH))

me_files = me_files[:SEQUENCE_LENGTH*(len(me_files)//SEQUENCE_LENGTH)]
me_files = np.array(me_files).reshape((len(me_files)//SEQUENCE_LENGTH, SEQUENCE_LENGTH))
labels = [0]*len(not_me_files) + [1]*len(me_files)
all_files = np.vstack((not_me_files,me_files))
all_files = zip(all_files, labels)

for ind, (sequence, label) in enumerate(all_files):
    sequence_for_test = []
    for fname in sequence:
        f = wave.open(fname, "rb")
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        f.close()
        rms_audio = np.sqrt(np.mean(data**2))
        if rms_audio != 0:
            dBFS = 10*np.log10(rms_audio/(2**15))
            scaled_speak = data * (10**((-10 - dBFS)/10))
        else:
            scaled_speak = data

        _, _, Sxx = signal.spectrogram(scaled_speak, fs=44100, nperseg=512, mode="magnitude")
        #Add tiny value to avoid 0
        scaled = 10*np.log10(Sxx+1e-9)
        sequence_for_test += [scaled]
    sequence_for_test = np.array(sequence_for_test).reshape((1,SEQUENCE_LENGTH, sequence_for_test[0].shape[0], sequence_for_test[0].shape[1], 1))


    pred = model.predict(sequence_for_test, batch_size=1)

    strg = "Prediction for sequence {}: {}, sigmoid out: {}".format(ind, pred, tf.keras.activations.sigmoid(pred))
    spaces = " "*(175 - len(strg))
    strg += spaces
    if label and pred > 0:
        successes += 1
        strg += "\t" + bcolors.OKGREEN + "PASS" + bcolors.ENDC
    elif label and pred <= 0:
        fails += 1
        failed_files += [(sequence, pred, label)]
        strg += "\t" + bcolors.FAIL + "FAIL" + bcolors.ENDC
    elif not label and pred > 0:
        fails += 1
        failed_files += [(sequence, pred, label)]
        strg += "\t" + bcolors.FAIL + "FAIL" + bcolors.ENDC
    else:
        successes += 1
        strg += "\t" + bcolors.OKGREEN + "PASS" + bcolors.ENDC
    print(strg)

print("Failed files:")
for seq, pred, label in failed_files:
    strg = ""
    strg += str(seq)
    strg += " "*(130 - len(strg))
    strg += "prediction: " + str(pred)
    strg += " "*(140 - len(strg))
    strg += "\tsigmoid prediction: {}".format(tf.keras.activations.sigmoid(pred))
    strg += " "*(170 - len(strg))
    strg += "correct: " + str(label)
    print(strg)
print("Total fails: {}\nTotal successes: {}\nPercentage: {}".format(fails, successes, successes/len(all_files)))
