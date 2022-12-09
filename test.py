import pyaudio
import wave
import time
import numpy as np
from scipy import signal
"""
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
"""
pa = pyaudio.PyAudio()

chunk_size = 1024
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
seconds = 5
print("recording now")
f1, f2 = 20, 21000
window = signal.firwin(3, [f1, f2], pass_zero=False, fs=44100)
def callback(in_data, frame_count, time_info, status_flags):
    in_data_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    #print("len of in data ", str(len(in_data)))
    #print("frame count ", str(frame_count))
    out_data = np.convolve(in_data_np, window, mode="same")
    #print("dims of np convert ", str(out_data.shape))
    #filtered = np.convolve(out_data, window, mode="same").astype(np.float32)
    #print("shape of filtered", str(filtered.shape))
    return (out_data.astype(np.int16).tobytes(), pyaudio.paContinue)

stream = pa.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk_size,
                    input=True,
                    output=True,
                    stream_callback=callback)
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
