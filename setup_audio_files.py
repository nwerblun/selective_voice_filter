import wave
import os
import shutil
from scipy import signal
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from PIL import Image
"""
http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/44.1kHz_16bit/
Start with chunking VoxAudio clips into 1 second clips
"""

"""
https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
"""

vox_roots = [
            "C:\\Users\\NWerblun\\Downloads\\mojomove411-20071206\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\chocoholic-20070524\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\chocoholic-20070523\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\calamity-20071011-poe\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\delibab-20071029\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\ttm-20071024\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\granthulbert-ar-01032007\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\ductapeguy-20080423-nau\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\starlite-20070613-fur1\\wav"
]
vox_dumps = [
            "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data\\mojomove411-20071206_dump", \
            "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data\\chocoholic-20070524_dump", \
            "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data\\chocoholic-20070523_dump", \
            "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data\\calamity-20071011-poe_dump", \
            "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data\\delibab-20071029_dump", \
            "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data\\ttm-20071024_dump", \
            "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data\\granthulbert-ar-01032007_dump", \
            "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data\\ductapeguy-20080423-nau_dump", \
            "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data\\starlite-20070613-fur1_dump"
]
nick_root = "C:\\Users\\NWerblun\\Downloads\\nick"
nick_dump = "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data\\nick_dump"
nick_test_root = "C:\\Users\\NWerblun\\Downloads\\nick_test"
nick_test_dump = "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\test_data\\nick_test_dump"
white_noise_dump = "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\noise_data\\white_noise"

FILE_LEN = 1

def resize_and_generate_spectrograms(root, fs, desired_fs, new_len=1):
    _, subdirs, filenames = next(os.walk(root))
    for f in filenames:
        if os.path.splitext(f)[1] == ".wav":
            _wav_to_spectrogram(root, f, fs, desired_fs, new_len)
    for s in subdirs:
        resize_and_generate_spectrograms(os.path.join(root, s), fs, desired_fs, new_len)

def add_noise_to_clips(root, scale=0.1):
    _, subdirs, filenames = next(os.walk(root))
    for fname in filenames:
        if os.path.splitext(fname)[1] == ".wav":
            f = wave.open(os.path.join(root, fname), "rb")
            params = f.getparams()
            audio_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)
            f.close()
            std = (10**(-10/10)) * (2**16)/2
            noise = np.random.normal(loc=0, scale=std, size=(len(audio_data),))
            prop = np.max(np.abs(audio_data)) / np.max(np.abs(noise))
            audio_data = audio_data + (prop * scale * noise)
            f = wave.open(os.path.join(root, fname), "wb")
            f.setparams(params)
            f.writeframes(audio_data.astype(np.int16).tobytes())
            f.close()
    for s in subdirs:
        add_noise_to_clips(os.path.join(root, s), scale)

def _wav_to_spectrogram(root, fname, fs, new_fs, new_len, win_size=256, overlap=128, nfft=256):
    f = wave.open(os.path.join(root, fname), "rb")
    fs = f.getframerate()
    nf = f.getnframes()
    ch = f.getnchannels()
    sw = f.getsampwidth()
    audio_data = np.frombuffer(f.readframes(nf), dtype=np.int16).astype(np.float32)
    if ch == 2:
        l = audio_data[::2]
        r = audio_data[1::2]
        audio_data = (l+r)/2
    elif ch > 2:
        raise ValueError("Cannot handle more than 2 channel audio")
    audio_data = signal.resample_poly(audio_data, new_fs, fs, window=3.7)
    #Drop any extras to make it a multiple of FS
    audio_data = audio_data[:new_fs*new_len*(len(audio_data)//(new_fs*new_len))]
    audio_data = audio_data.reshape(len(audio_data)//(new_fs*new_len), new_fs*new_len)
    for ind, clip in enumerate(audio_data):
        filtered = signal.lfilter(np.array([1,-0.68]), np.array([1]), clip)
        f, t, Sxx = signal.spectrogram(filtered, fs=new_fs, nperseg=win_size, noverlap=overlap, window="blackman", nfft=nfft, mode="magnitude", scaling="spectrum")
        normer = LogNorm(vmin=Sxx.max()*5e-4, vmax=Sxx.max(), clip=True)
        sm = cm.ScalarMappable(norm=normer, cmap="magma")
        x = Image.fromarray(sm.to_rgba(np.flipud(Sxx), bytes=True))
        x.save(os.path.join(root, os.path.splitext(fname)[0]+"_"+str(ind)+".png"))
    os.remove(os.path.join(root, fname))

def make_offset_clip(file_root, start, stop):
    f = wave.open(file_root, "rb")
    fs = f.getframerate()
    nf = f.getnframes()
    ch = f.getnchannels()
    sw = f.getsampwidth()
    audio_data = np.frombuffer(f.readframes(nf), dtype=np.int16).astype(np.float32)
    f.close()
    audio_data = audio_data[start:stop]
    f = wave.open(os.path.splitext(file_root)[0]+"_sp.wav", "wb")
    f.setparams((ch, sw, fs, len(audio_data), "NONE", "not compressed"))
    f.writeframes(audio_data.astype(np.int16).tobytes())
    f.close()

def normalize_audio_volume(root, rms_in_dB=-10):
    _, subdirs, filenames = next(os.walk(root))
    for fname in filenames:
        if os.path.splitext(fname)[1] == ".wav":
            f = wave.open(os.path.join(root, fname), "rb")
            params = f.getparams()
            audio_data = 1e-9 + np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)
            f.close()
            rms = np.sqrt(np.mean(audio_data**2))
            #assume 16 bit samples, add functionality some other time maybe
            #hardcode the max possible sample size (2^16)/2 to save some time.
            dBFS = 10*np.log10(rms/32768.0)
            gain = 10**((rms_in_dB - dBFS)/10)
            audio_data *= gain
            f = wave.open(os.path.join(root, fname), "wb")
            f.setparams(params)
            f.writeframes(audio_data.astype(np.int16).tobytes())
            f.close()
    for s in subdirs:
        normalize_audio_volume(os.path.join(root, s), rms_in_dB)

print("Working on voice clips...")
for dir, dump_dir in zip(vox_roots, vox_dumps):
    shutil.copytree(dir, dump_dir, dirs_exist_ok=True)
    add_noise_to_clips(dump_dir, 0.02)
    normalize_audio_volume(dump_dir, -10)
    resize_and_generate_spectrograms(dump_dir, 44100, 16000, FILE_LEN)

shutil.copytree(nick_root, nick_dump, dirs_exist_ok=True)
add_noise_to_clips(nick_dump, 0.02)
normalize_audio_volume(nick_dump, -10)
resize_and_generate_spectrograms(nick_dump, 44100, 16000, FILE_LEN)

shutil.copytree(nick_test_root, nick_test_dump, dirs_exist_ok=True)
add_noise_to_clips(nick_test_dump, 0.02)
normalize_audio_volume(nick_test_dump, -10)
resize_and_generate_spectrograms(nick_test_dump, 44100, 16000, FILE_LEN)

print("moving a few examples to test directory...")
for dump_dir in vox_dumps:
    os.makedirs(os.path.join("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\test_data", os.path.split(dump_dir)[1]), exist_ok=True)
    _, _, filenames = next(os.walk(dump_dir))
    to_move = filenames[-int(len(filenames) * 0.2):]
    for fname in to_move:
        shutil.move(os.path.join(dump_dir, fname), os.path.join("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\test_data", os.path.split(dump_dir)[1], fname))


_, _, filenames = next(os.walk(nick_dump))
os.makedirs(os.path.join("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\test_data", os.path.split(nick_dump)[1]), exist_ok=True)
to_move = filenames[-int(len(filenames) * 0.2):]
for fname in to_move:
    shutil.move(os.path.join(nick_dump, fname), os.path.join("C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\test_data", os.path.split(nick_dump)[1], fname))
