import pyaudio
import wave
import os
import time
import numpy as np
from scipy import signal
import tensorflow as tf
from tensorflow import keras
import threading

cls = lambda : os.system("cls")
pa = pyaudio.PyAudio()
model = keras.models.load_model("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\model.h5")
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
seconds = 1
feedthrough_chunk_size_read = 1024 # ~23ms of data
feedthrough_chunk_size_write = 512
#/2 because it's signed and we want only >0 vals since we compare to RMS
max_possible = (2**16)/2
f1, f2 = 20, 21000
#Filter with no gain
window = signal.firwin(3, [f1, f2], pass_zero=False, fs=fs)

GO = True
dBFS = 0
PREDICTION = -float("inf")

#mag = 10**(dB/10) * scale. 0.1 comes from using -10dB, 2**15 = 2**16/2
#the comma is not a mistake. Queue store 1 sec. of data init to white noise
QUEUE = np.random.normal(loc=0, scale=(0.1 * (2**15)), size=(fs,)).astype(np.float32)
PRED_THRESH = 0.97
queue_lock = threading.Lock()

def callback_read(in_data, frame_count, time_info, status_flags):
    global QUEUE
    in_data_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    with queue_lock:
        QUEUE = np.append(QUEUE[feedthrough_chunk_size:], in_data_np)
    return (None, pyaudio.paContinue)

def callback_write(in_data, frame_count, time_info, status_flags):
    global QUEUE
    global dBFS
    global PREDICTION
    with queue_lock:
        temp_queue = QUEUE[:]
    rms_audio = np.sqrt(np.mean(temp_queue**2))
    if rms_audio != 0:
        dBFS = 10*np.log10(rms_audio/32768) #again, assuming 16 bits
        scaled_speak = temp_queue * (10**((-10 - dBFS)/10))
    else:
        scaled_speak = temp_queue
    fft = np.fft.fft(scaled_speak)
    #keep only pos half of mag. spec.
    fft = np.abs(fft).astype(np.float32)[:len(fft)//2]
    fft = fft.reshape((1,-1))
    PREDICTION = float(tf.keras.activations.sigmoid(model.predict(fft, batch_size=1, verbose=0)))
    if PREDICTION == 1:
        return (temp_queue[-feedthrough_chunk_size_write:].astype(np.int16).tobytes(), pyaudio.paContinue)
    else:
        return(np.zeros((feedthrough_chunk_size_write,)).astype(np.int16).tobytes(), pyaudio.paContinue)


def make_prediction(stop):
    global PREDICTION
    while True:
        if stop():
            break
        with queue_lock:
            temp_queue = QUEUE[:]
        rms_audio = np.sqrt(np.mean(temp_queue**2))
        if rms_audio != 0:
            db = 10*np.log10(rms_audio/(2**15))
            scaled_speak = temp_queue * (10**((-10 - db)/10))
        else:
            scaled_speak = temp_queue
        fft = np.fft.fft(temp_queue)
        #keep only pos half of mag. spec.
        fft = np.abs(fft).astype(np.float32)[:len(fft)//2]
        fft = fft.reshape((1,-1))
        PREDICTION = float(tf.keras.activations.sigmoid(model.predict(fft, batch_size=1, verbose=0)))

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
                "\nCurrent Prediction (1 sec. delayed): {:2.2f}".format(PREDICTION)
        )
        time.sleep(0.05)


"""
Changes when restarting computer
Write to VB cable input, and it gets carried to VB cable output. Use VB cable output as the discord input
{'index': 11, 'structVersion': 2, 'name': 'CABLE Input (VB-Audio Virtual C', 'hostApi': 0, 'maxInputChannels': 0, 'maxOutputChannels': 8,
'defaultLowInputLatency': 0.09, 'defaultLowOutputLatency': 0.09, 'defaultHighInputLatency': 0.18,
'defaultHighOutputLatency': 0.18, 'defaultSampleRate': 44100.0}
"""
stream_write = pa.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=feedthrough_chunk_size_write,
                    input=False,
                    output=True,
                    output_device_index=11,
                    stream_callback=callback_write)

stream_read = pa.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=feedthrough_chunk_size_read,
                    input=True,
                    output=False,
                    stream_callback=callback_read)

try:
    stop_threads = False
    mon = threading.Thread(target=status_print, args=(lambda : stop_threads,), daemon=True)
    #predictor = threading.Thread(target=make_prediction, args=(lambda : stop_threads,), daemon=True)
    mon.start()
    #predictor.start()
    while True:
        time.sleep(0.2)
except KeyboardInterrupt:
    print("Killing Process")
    stop_threads = True
    stream_write.stop_stream()
    stream_write.close()
    stream_read.stop_stream()
    stream_read.close()
    pa.terminate()
