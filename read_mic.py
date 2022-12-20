import pyaudio
import wave
import os
import time
import numpy as np
from scipy import signal
import tensorflow as tf
from tensorflow import keras
import threading


"""
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
"""

cls = lambda : os.system("cls")
pa = pyaudio.PyAudio()
model = keras.models.load_model("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\model.h5")
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
seconds = 1
feedthrough_chunk_size = 512 # ~11.6ms of data
#/2 because it's signed and we want only >0 vals since we compare to RMS
max_possible = (2**16)/2
f1, f2 = 20, 21000
#Filter with no gain
window = signal.firwin(3, [f1, f2], pass_zero=False, fs=fs)

GO = True
dBFS = 0
PREDICTION = -float("inf")
QUEUE = np.zeros((fs,)) #the comma is not a mistake
PRED_THRESH = 1000
queue_lock = threading.Lock()

def callback(in_data, frame_count, time_info, status_flags):
    global QUEUE
    global dBFS
    in_data_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(in_data_np**2))
    dBFS = 10*np.log10(rms/max_possible)
    with queue_lock:
        QUEUE = np.append(QUEUE[feedthrough_chunk_size:], in_data_np)
    in_data_np_filt = np.convolve(in_data_np, window, mode="same")
    if PREDICTION >= PRED_THRESH:
        return (in_data_np_filt.astype(np.int16).tobytes(), pyaudio.paContinue)
    else:
        return (np.zeros(in_data_np.shape).astype(np.int16).tobytes(), pyaudio.paContinue)

def make_prediction(stop):
    global PREDICTION
    while True:
        if stop():
            break
        with queue_lock:
            temp_queue = QUEUE[:]
        fft = np.fft.fft(temp_queue)
        #keep only pos half of mag. spec.
        fft = np.abs(fft).astype(np.float32)[:len(fft)//2]
        fft = fft.reshape((1,-1))
        PREDICTION = float(model.predict(fft, batch_size=1, verbose=0))

def thresh_monitor(stop):
    strt = time.time()
    silent_time_thresh = 3000 #3 seconds
    triggered = False
    while True:
        if stop():
            break

        if silent_time_thresh - strt > 0 and dBFS >= -9:
            PRED_THRESH = -float("inf")
        else:
            PRED_THRESH = 4000

        if dBFS >= -9:
            strt = float("inf")
            triggered = False
        elif not triggered and dBFS <= -20:
            triggered = True
            strt = time.time()



def status_print(stop):
    #Don't need globals because we are just reading them
    while True:
        if stop():
            break
        cls()
        print("Press CTRL+C To End", \
                "\ndBFS Reading: {:2.2f}".format(dBFS), \
                "\nCurrent Prediction (1 sec. delayed): {:2.2f}\nSigmoid Prediction (1 sec. delayed): {:2.2f}".format(PREDICTION, tf.keras.activations.sigmoid(PREDICTION))
        )
        if PREDICTION >= PRED_THRESH:
            print("Mic Status: Active")
        else:
            print("Mic Status: Muted")
        time.sleep(0.045)


"""
Write to VB cable input, and it gets carried to VB cable output. Use VB cable output as the discord input
{'index': 10, 'structVersion': 2, 'name': 'CABLE Input (VB-Audio Virtual C', 'hostApi': 0, 'maxInputChannels': 0, 'maxOutputChannels': 8,
'defaultLowInputLatency': 0.09, 'defaultLowOutputLatency': 0.09, 'defaultHighInputLatency': 0.18,
'defaultHighOutputLatency': 0.18, 'defaultSampleRate': 44100.0}
"""
stream_write = pa.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=feedthrough_chunk_size,
                    input=True,
                    output=True,
                    output_device_index=10,
                    stream_callback=callback)

try:
    stop_threads = False
    mon = threading.Thread(target=status_print, args=(lambda : stop_threads,), daemon=True)
    predictor = threading.Thread(target=make_prediction, args=(lambda : stop_threads,), daemon=True)
    mon.start()
    predictor.start()
    while True:
        time.sleep(0.2)
except KeyboardInterrupt:
    print("Killing Process")
    stop_threads = True
    stream_write.stop_stream()
    stream_write.close()
    pa.terminate()
