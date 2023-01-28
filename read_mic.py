import pyaudio
import time
import os
import numpy as np
from scipy import signal
from tensorflow import keras
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from PIL import Image
from multiprocessing import Process, Value
import tkinter as tk
from tkinter import ttk

pa = pyaudio.PyAudio()
model = keras.models.load_model("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\model.h5")
sample_format = pyaudio.paInt16
channels = 1
fs = 16000
feedthrough_chunk_size = 2048
PREDICTION = -float("inf")
#Compute once and reuse
time_chunks = int((16000/256) + ((16000-128)/256))
#Two bytes per sample, so need double the length. 1 second + 1 buffer
QUEUE_fs = bytearray(88200+4096)
MAX_MOMENTUM = 5
last_n_pred = -5
last_process_time = 0

stream = None
stop = 0
root = tk.Tk()
output_devices = {}
FILTER_ON = True
MIC_MUTED = False

def callback_both(in_data, frame_count, time_info, status_flags):
    global QUEUE_fs
    global PREDICTION
    global last_n_pred
    global last_process_time

    p0 = time.perf_counter_ns()
    if MIC_MUTED:
        last_process_time = time.perf_counter_ns() - p0
        return (np.zeros(4096,).tobytes(), pyaudio.paContinue)

    if not FILTER_ON:
        last_process_time = time.perf_counter_ns() - p0
        return (in_data, pyaudio.paContinue)

    #Cannot do this in one line with byte arrays
    QUEUE_fs = QUEUE_fs[4096:]
    QUEUE_fs.extend(in_data)
    np_data = np.frombuffer(QUEUE_fs[:88200], dtype=np.int16).astype(np.float32)
    audio_data = signal.resample_poly(np_data, 16000, 44100, window=3.7)
    filtered = signal.filtfilt(np.array([1,-0.68]), np.array([1]), audio_data)
    _, _, Sxx = signal.spectrogram(filtered, fs=16000, nperseg=256, noverlap=128, window="blackman", nfft=256, mode="magnitude", scaling="spectrum")
    normer = LogNorm(vmin=Sxx.max()*5e-4, vmax=Sxx.max(), clip=True)
    sm = cm.ScalarMappable(norm=normer, cmap="magma")
    rgb = Image.fromarray(sm.to_rgba(np.flipud(Sxx), bytes=True))
    rgb = np.array(list(rgb.convert("RGB").getdata()))
    rgb = rgb.reshape((
        1,
        256//2+1,
        time_chunks,
        3
    ))
    PREDICTION = model(rgb, training=False)
    sig = keras.activations.sigmoid(PREDICTION)
    if sig >= 0.97:
        last_n_pred = last_n_pred + 1 if last_n_pred < MAX_MOMENTUM else MAX_MOMENTUM
        p12 = time.perf_counter_ns()
        last_process_time = p12-p0
        if last_n_pred >= 0:
            return (in_data, pyaudio.paContinue)
        else:
            return (np.zeros(4096,).tobytes(), pyaudio.paContinue)
    else:
        last_n_pred = last_n_pred - 1 if last_n_pred > -MAX_MOMENTUM else -MAX_MOMENTUM
        p12 = time.perf_counter_ns()
        last_process_time = p12-p0
        if last_n_pred <= 0:
            return (np.zeros(4096,).tobytes(), pyaudio.paContinue)
        else:
            return (in_data, pyaudio.paContinue)

def quit():
    global stop
    stop = 1
    print("Killing Process")
    stream.stop_stream()
    stream.close()
    pa.terminate()
    return

def process_time_update_loop(stop, label):
    next = None
    if stop():
        if next:
            root.after_cancel(next)
        return
    else:
        label["text"] = "Last Processing Time: {:.3f}ms".format(last_process_time*1e-6)
        next = root.after(100, process_time_update_loop, stop, label)
    return

def last_prediction_update_loop(stop, label, label2):
    pred = float(PREDICTION)
    next = None
    if stop():
        if next:
            root.after_cancel(next)
        return
    else:
        label["text"] = "Last Prediction: {:.3f}".format(pred)
        label2["text"] = "Sigmoid of Last Prediction: {:.3f}".format(keras.activations.sigmoid(pred))
        next = root.after(100, last_prediction_update_loop, stop, label, label2)
    return

def momentum_update_loop(stop, label):
    next = None
    if stop():
        if next:
            root.after_cancel(next)
        return
    else:
        label["text"] = "Current Momentum: {}".format(last_n_pred)
        next = root.after(100, momentum_update_loop, stop, label)
    return

def open_new_stream_close_old(output_index):
    global stream
    if stream:
        stream.stop_stream()
        stream.close()
    stream = pa.open(format=sample_format,
                channels=channels,
                rate=44100,
                frames_per_buffer=feedthrough_chunk_size,
                input=True,
                output=True,
                output_device_index=output_index,
                stream_callback=callback_both)
    return

def device_change(event):
    name = combo.get()
    open_new_stream_close_old(output_devices[name])
    return

def momentum_change(event):
    global MAX_MOMENTUM
    MAX_MOMENTUM = event.widget.get()
    return

def press_power_button():
    global FILTER_ON
    if FILTER_ON:
        new_im = tk.PhotoImage(file = r".\images\red_pwr.png")
        enable_button.configure(image=new_im)
        enable_button.photo = new_im #need this to avoid GC removing the link
    else:
        new_im = tk.PhotoImage(file = r".\images\green_pwr.png")
        enable_button.configure(image=new_im) #need this to avoid GC removing the link
        enable_button.photo = new_im
    FILTER_ON = not FILTER_ON
    return

def press_mute_button():
    global MIC_MUTED
    if MIC_MUTED:
        new_im = tk.PhotoImage(file = r".\images\mic_on.png")
        mic_en_button.configure(image=new_im)
        mic_en_button.photo = new_im #need this to avoid GC removing the link
    else:
        new_im = tk.PhotoImage(file = r".\images\mic_off.png")
        mic_en_button.configure(image=new_im)
        mic_en_button.photo = new_im #need this to avoid GC removing the link
    MIC_MUTED = not MIC_MUTED
    return

"""
Changes when restarting computer
Write to VB cable input, and it gets carried to VB cable output. Use VB cable output as the discord input
{'index': 11, 'structVersion': 2, 'name': 'CABLE Input (VB-Audio Virtual C', 'hostApi': 0, 'maxInputChannels': 0, 'maxOutputChannels': 8,
'defaultLowInputLatency': 0.09, 'defaultLowOutputLatency': 0.09, 'defaultHighInputLatency': 0.18,
'defaultHighOutputLatency': 0.18, 'defaultSampleRate': 44100.0}
"""

if __name__ == "__main__":
    try:
        #root.geometry('500x500')
        frm = tk.Frame(root, relief=tk.RAISED, borderwidth=1)
        frm.grid(column=0, row=0, padx=3, pady=1)

        frm_title = tk.Label(frm, text="Debug Info")
        frm_title.grid(column=0, row=0, sticky="n")

        process_time_label = tk.Label(frm, text="Last process time: {}".format(last_process_time))
        process_time_label.grid(column=0, row=1, sticky="w")

        momentum_label = tk.Label(frm, text="Momentum: {}".format(last_n_pred))
        momentum_label.grid(column=0, row=2, sticky="w")

        pred_label = tk.Label(frm, text="Last Prediction: {}".format(PREDICTION))
        pred_label.grid(column=0, row=3, sticky="w")

        sig_pred_label = tk.Label(frm, text="Sigmoid of Last Prediction: {}".format(keras.activations.sigmoid(PREDICTION)))
        sig_pred_label.grid(column=0, row=4, sticky="w")

        dropdown_frm = tk.Frame(root, relief=tk.RAISED, borderwidth=1)
        dropdown_frm.grid(column=1, row=0, padx=3, pady=1)

        dropdown_title_label = tk.Label(dropdown_frm, text="Output Device Select")
        dropdown_title_label.grid(column=0, row=0, sticky="n")

        info = pa.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        default_choice = ""
        for i in range(0, numdevices):
            if (pa.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels')) > 0:
                name = pa.get_device_info_by_host_api_device_index(0, i).get('name')
                if "VB" in name:
                    default_choice = name
                output_devices[name] = i

        combo = ttk.Combobox(
            dropdown_frm,
            state="readonly",
            values = list(output_devices.keys())
        )
        combo.bind("<<ComboboxSelected>>", device_change)
        combo.grid(column=0, row=1, sticky="n")
        combo.set(default_choice)

        stream = pa.open(format=sample_format,
                            channels=channels,
                            rate=44100,
                            frames_per_buffer=feedthrough_chunk_size,
                            input=True,
                            output=True,
                            output_device_index=output_devices[default_choice],
                            stream_callback=callback_both)

        slider_frame = tk.Frame(root)
        slider_frame.grid(column=0, row=1, columnspan=2, padx=2, pady=0, sticky="n")
        slider_label = tk.Label(slider_frame, text="Max Momentum")
        slider_label.grid(column=0, row=0, pady=1, sticky="n")

        slider_bar = tk.Scale(slider_frame, from_=0, to=50, orient="horizontal", length=350)
        slider_bar.grid(column=0, row=1, pady=1, sticky="n")
        slider_bar.set(5)
        slider_bar.bind("<ButtonRelease-1>", momentum_change)

        buttons_frame = tk.Frame(root)
        buttons_frame.grid(column=0, row=2, columnspan=2, padx=3, pady=1, sticky="n")

        pwr_on = tk.PhotoImage(file = r".\images\green_pwr.png")
        enable_button = ttk.Button(buttons_frame, image=pwr_on, command=press_power_button)
        enable_button.grid(column=0, row=0, padx=15, pady=5)

        mic_on = tk.PhotoImage(file = r".\images\mic_on.png")
        mic_en_button = ttk.Button(buttons_frame, image=mic_on, command=press_mute_button)
        mic_en_button.grid(column=1, row=0, padx=15, pady=5)

        process_time_update_loop(lambda : stop, process_time_label)
        momentum_update_loop(lambda : stop, momentum_label)
        last_prediction_update_loop(lambda : stop, pred_label, sig_pred_label)

        root.resizable(0, 0)
        root.title("Selective Voice Filter")
        root.mainloop()
        quit()
    except KeyboardInterrupt:
        print("Killing Process")
        stop = 1
        root.destroy()
        stream.stop_stream()
        stream.close()
        pa.terminate()
