import pyaudio
import time
import numpy as np
from scipy import signal
#import tensorflow.compat.v1 as tf
from tensorflow import keras
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from PIL import Image

pa = pyaudio.PyAudio()
model = keras.models.load_model("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\model.h5")
sample_format = pyaudio.paInt16
channels = 1
fs = 16000
feedthrough_chunk_size = 2048

dBFS = 0
PREDICTION = -float("inf")
#Compute once and reuse
time_chunks = int((16000/256) + ((16000-128)/256))
#Two bytes per sample, so need double the length. 1 second + 1 buffer
QUEUE_fs = bytearray(88200+4096)


# Load keras model as a TF v1 graph with no eager execution
# see here: https://ksingh7.medium.com/part-iii-convert-keras-model-to-tensorflow-frozen-graph-model-a6aa6b1aaeee
# sess=tf.InteractiveSession()
# frozen_graph="./frozen_model.pb"
# with tf.gfile.GFile(frozen_graph, "rb") as f:
#       graph_def = tf.GraphDef()
#       graph_def.ParseFromString(f.read())
# sess.graph.as_default()
# tf.import_graph_def(graph_def)
# input_tensor = sess.graph.get_tensor_by_name("x:0")
# output_tensor = sess.graph.get_tensor_by_name("Identity:0")

def callback_both(in_data, frame_count, time_info, status_flags):
    global QUEUE_fs
    global PREDICTION
    # p0 = time.perf_counter_ns()
    #Cannot do this in one line with byte arrays
    QUEUE_fs = QUEUE_fs[4096:]
    QUEUE_fs.extend(in_data)
    np_data = np.frombuffer(QUEUE_fs[:88200], dtype=np.int16).astype(np.float32)
    # p1 = time.perf_counter_ns()
    audio_data = signal.resample_poly(np_data, 16000, 44100, window=3.7)
    # p2 = time.perf_counter_ns()
    filtered = signal.filtfilt(np.array([1,-0.68]), np.array([1]), audio_data)
    # p3 = time.perf_counter_ns()
    _, _, Sxx = signal.spectrogram(filtered, fs=16000, nperseg=256, noverlap=128, window="blackman", nfft=256, mode="magnitude", scaling="spectrum")
    # p4 = time.perf_counter_ns()
    normer = LogNorm(vmin=Sxx.max()*5e-4, vmax=Sxx.max(), clip=True)
    # p5 = time.perf_counter_ns()
    sm = cm.ScalarMappable(norm=normer, cmap="magma")
    # p6 = time.perf_counter_ns()
    rgb = Image.fromarray(sm.to_rgba(np.flipud(Sxx), bytes=True))
    # p7 = time.perf_counter_ns()
    rgb = np.array(list(rgb.convert("RGB").getdata()))
    # p8 = time.perf_counter_ns()
    rgb = rgb.reshape((
        1,
        256//2+1,
        time_chunks,
        3
    ))
    # p9 = time.perf_counter_ns()
    PREDICTION = model(rgb, training=False)
    # PREDICTION = sess.run(output_tensor, {'x:0': rgb})
    # p10 = time.perf_counter_ns()
    sig = tf.keras.activations.sigmoid(PREDICTION)
    # p11 = time.perf_counter_ns()
    # print("\n"*4+"PRED: {}\nSIGM: {}".format(PREDICTION, sig))
    # print("Queue ops: {:2.2f}ms".format(1e-6 * (p1-p0)))
    # print("Resample: {:2.2f}ms".format(1e-6 * (p2-p1)))
    # print("Filter: {:2.2f}ms".format(1e-6 * (p3-p2)))
    # print("Spectrogram: {:2.2f}ms".format(1e-6 * (p4-p3)))
    # print("Normalize: {:2.2f}ms".format(1e-6 * (p5-p4)))
    # print("Cmap: {:2.2f}ms".format(1e-6 * (p6-p5)))
    # print("To image: {:2.2f}ms".format(1e-6 * (p7-p6)))
    # print("To rgb array: {:2.2f}ms".format(1e-6 * (p8-p7)))
    # print("Reshape: {:2.2f}ms".format(1e-6 * (p9-p8)))
    # print("Predict: {:2.2f}ms".format(1e-6 * (p10-p9)))
    # print("Sigmoid: {:2.2f}ms".format(1e-6 * (p11-p10)))
    if sig >= 0.97:
        # p12 = time.perf_counter_ns()
        # print("Compare: {:2.2f}ms".format(1e-6 * (p12-p11)))
        # print("Total: {:2.2f}ms".format(1e-6 * (p12-p0)))
        return (in_data, pyaudio.paContinue)
    else:
        # p12 = time.perf_counter_ns()
        # print("Compare: {}ms".format(1e-6 * (p12-p11)))
        # print("Total with prints: {:2.2f}ms".format(1e-6 * (p12-p0)))
        return (np.zeros(4096,).tobytes(), pyaudio.paContinue)

"""
Changes when restarting computer
Write to VB cable input, and it gets carried to VB cable output. Use VB cable output as the discord input
{'index': 11, 'structVersion': 2, 'name': 'CABLE Input (VB-Audio Virtual C', 'hostApi': 0, 'maxInputChannels': 0, 'maxOutputChannels': 8,
'defaultLowInputLatency': 0.09, 'defaultLowOutputLatency': 0.09, 'defaultHighInputLatency': 0.18,
'defaultHighOutputLatency': 0.18, 'defaultSampleRate': 44100.0}
"""
stream = pa.open(format=sample_format,
                    channels=channels,
                    rate=44100,
                    frames_per_buffer=feedthrough_chunk_size,
                    input=True,
                    output=True,
                    output_device_index=11,
                    stream_callback=callback_both)

try:
    while True:
        time.sleep(0.6)
except KeyboardInterrupt:
    print("Killing Process")
    stream.stop_stream()
    stream.close()
    pa.terminate()
