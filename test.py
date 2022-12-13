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
window = signal.firwin(3, [f1, f2], pass_zero=False, fs=fs)
def callback(in_data, frame_count, time_info, status_flags):
    in_data_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    #print("len of in data ", str(len(in_data)))
    #print("frame count ", str(frame_count))
    out_data = np.convolve(in_data_np, window, mode="same")
    #print("dims of np convert ", str(out_data.shape))
    #filtered = np.convolve(out_data, window, mode="same").astype(np.float32)
    #print("shape of filtered", str(filtered.shape))
    return (out_data.astype(np.int16).tobytes(), pyaudio.paContinue)

"""
Write to VB cable input, and it gets carried to VB cable output. Use VB cable output as the discord input
{'index': 10, 'structVersion': 2, 'name': 'CABLE Input (VB-Audio Virtual C', 'hostApi': 0, 'maxInputChannels': 0, 'maxOutputChannels': 8,
'defaultLowInputLatency': 0.09, 'defaultLowOutputLatency': 0.09, 'defaultHighInputLatency': 0.18,
'defaultHighOutputLatency': 0.18, 'defaultSampleRate': 44100.0}
"""
stream = pa.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk_size,
                    input=True,
                    output=True,
                    output_device_index=10,
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
