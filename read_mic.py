import pyaudio
import wave
import time
import numpy as np
from scipy import signal
import tensorflow as tf
from tensorflow import keras


"""
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
"""
pa = pyaudio.PyAudio()
model = keras.models.load_model("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\model.h5")
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
seconds = 1
print("recording now")
f1, f2 = 20, 21000
window = signal.firwin(3, [f1, f2], pass_zero=False, fs=fs)
GO = 1
def callback(in_data, frame_count, time_info, status_flags):
    #skip filtering for now.
    #in_data_np_filt = np.convolve(in_data_np, window, mode="same")
    global GO
    if GO:
        return (in_data, pyaudio.paContinue)
    else:
        in_data_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        return (np.zeros(in_data_np.shape).astype(np.int16).tobytes(), pyaudio.paContinue)

def callback2(in_data, frame_count, time_info, status_flags):
    global GO
    in_data_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    fft = np.fft.fft(in_data_np)
    #keep only pos half of mag. spec.
    fft = np.abs(fft).astype(np.float32)[:len(fft)//2]
    fft = fft.reshape((1,-1))
    temp_last_state = GO
    GO = model.predict(fft, batch_size=1, verbose=0) > 0.98
    if not temp_last_state and GO:
        print("Now outputting voice")
    elif temp_last_state and not GO:
        print("Muting voice stream")
    return (in_data, pyaudio.paContinue)

"""
Write to VB cable input, and it gets carried to VB cable output. Use VB cable output as the discord input
{'index': 10, 'structVersion': 2, 'name': 'CABLE Input (VB-Audio Virtual C', 'hostApi': 0, 'maxInputChannels': 0, 'maxOutputChannels': 8,
'defaultLowInputLatency': 0.09, 'defaultLowOutputLatency': 0.09, 'defaultHighInputLatency': 0.18,
'defaultHighOutputLatency': 0.18, 'defaultSampleRate': 44100.0}
"""
stream_write = pa.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=512,
                    input=True,
                    output=True,
                    output_device_index=10,
                    stream_callback=callback)

stream_read_only = pa.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=fs*seconds,
                    input=True,
                    output=False,
                    stream_callback=callback2)

try:
    print("* echoing")
    print("Press CTRL+C to stop")
    while True:
        time.sleep(0.2)
    print("* done echoing")
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    pa.terminate()
