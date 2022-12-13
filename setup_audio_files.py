import wave
import os
import shutil
from scipy import signal
import numpy as np
"""
http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/44.1kHz_16bit/
Start with chunking VoxAudio clips into 1 second clips
"""


"""
https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
1. Resample Kaggle set from 16kHz to 44.1kHz
2. Transform non-1sec clips into 1 second clips
"""
kaggle_root = "C:\\Users\\NWerblun\\Downloads\\16000_pcm_speeches"
kaggle_dump = "C:\\Users\\NWerblun\\Downloads\\16000_pcm_speeches_dump"
vox_roots = ["C:\\Users\\NWerblun\\Downloads\\mojomove411-20071206\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\chocoholic-20070524\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\chocoholic-20070523\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\calamity-20071011-poe\\wav"]
vox_dumps = ["C:\\Users\\NWerblun\\Downloads\\mojomove411-20071206_dump\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\chocoholic-20070524_dump\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\chocoholic-20070523_dump\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\calamity-20071011-poe_dump\\wav"]
nick_root = "C:\\Users\\NWerblun\\Downloads\\nick"
nick_dump = "C:\\Users\\NWerblun\\Downloads\\nick_dump"


def resample_and_resize(root):
    _, subdirs, filenames = next(os.walk(root))
    for f in filenames:
        if os.path.splitext(f)[1] == ".wav":
            up_dn_sample_and_split(root, f, 44100)
    for s in subdirs:
        resample_and_resize(os.path.join(root, s))

def up_dn_sample_and_split(root, fname, new_fs, file_len=1):
    #File len is desired length in seconds. It will be split up if longer.
    _up_dn_sample(root, fname, new_fs)
    _split_file(root, fname, file_len)

def _split_file(root, fname, file_len):
    f = wave.open(os.path.join(root, fname), "rb")
    fs = f.getframerate()
    nf = f.getnframes()
    if (nf/fs == file_len):
        #print("File length ok, skipping")
        return
    ch = f.getnchannels()
    sw = f.getsampwidth()
    audio_data = np.frombuffer(f.readframes(nf), dtype=np.int16).astype(np.float32)
    if ch == 2:
        l = audio_data[::2]
        r = audio_data[1::2]
        audio_data = (l+r)/2
    elif ch > 2:
        raise ValueError("Cannot handle more than 2 channel audio")
    audio_data = audio_data.reshape(1, nf)
    f.close()

    extra_zeros = 0
    to_pad = 0
    if nf/fs < file_len:
        print("need to zero pad to reach desired file length")
        extra_zeros = (file_len * fs) - fs*(nf//fs)
    if int((nf % fs) > 0):
        print("padding a few zeros to reach a multiple of the samp. freq.")
        to_pad = fs - (nf%fs)
    padded = np.zeros((1, nf+to_pad+extra_zeros))
    padded[:,:nf] = audio_data
    audio_data = padded
    #can't use nf here because len may have changed if it was padded.
    #Also it's a matrix so row 1 is element 0
    audio_data = np.reshape(audio_data, (len(audio_data[0])//fs, fs))
    #write each row as a new file
    for ind, row in enumerate(audio_data):
        new_fname = os.path.splitext(fname)[0] + "_" + str(ind) + os.path.splitext(fname)[1]
        f = wave.open(os.path.join(root, new_fname), "wb")
        f.setparams((1, sw, fs, len(row), "NONE", "not compressed"))
        f.writeframes(row.astype(np.int16).tobytes())
        f.close()
    os.remove(os.path.join(root, fname))

def _up_dn_sample(root, fname, new_fs):
    f = wave.open(os.path.join(root, fname), "rb")
    old_fs = f.getframerate()
    f.close()
    if old_fs >= new_fs:
        _downsample(root, fname, new_fs)
    else:
        _upsample(root, fname, new_fs)

def _downsample(root, fname, new_fs):
    f = wave.open(os.path.join(root, fname), "rb")
    channels = f.getnchannels()
    old_num_frames, old_framerate, old_samp_width = f.getnframes(), f.getframerate(), f.getsampwidth()

    tp = np.int16
    if old_samp_width == 1:
        tp = np.int8
    elif old_samp_width == 2:
        tp = np.int16

    new_samp_width = old_samp_width
    audio_data = np.frombuffer(f.readframes(old_num_frames), dtype=tp).astype(np.float32)
    if channels == 2:
        #Split l/r channels
        new_l = signal.resample_poly(audio_data[::2], new_fs, old_framerate, window=3.7)
        new_r = signal.resample_poly(audio_data[1::2], new_fs, old_framerate, window=3.7)
        new_audio_data = np.zeros(audio_data.shape)
        new_audio_data[::2] = new_l
        new_audio_data[1::2] = new_r
    elif ch > 2:
        raise ValueError("Cannot handle more than 2 channel audio")
    else:
        new_audio_data = signal.resample_poly(audio_data, new_fs, old_framerate, window=3.7)
    new_len = len(new_audio_data)
    new_audio_data = new_audio_data.astype(np.int16).tobytes()
    f.close()

    f = wave.open(os.path.join(root, fname), "wb")
    f.setparams((channels, new_samp_width, new_fs, new_len, "NONE", "not compressed"))
    f.writeframes(new_audio_data)
    f.close()

def _upsample(root, fname, new_fs):
    f = wave.open(os.path.join(root, fname), "rb")
    channels = f.getnchannels()
    old_num_frames, old_framerate, old_samp_width = f.getnframes(), f.getframerate(), f.getsampwidth()

    tp = np.int16
    if old_samp_width == 1:
        tp = np.int8
    elif old_samp_width == 2:
        tp = np.int16

    new_samp_width = old_samp_width
    audio_data = np.frombuffer(f.readframes(old_num_frames), dtype=tp).astype(np.float32)
    if channels == 2:
        #Split l/r channels
        new_l = signal.resample_poly(audio_data[::2], new_fs, old_framerate, window=3.7)
        new_r = signal.resample_poly(audio_data[1::2], new_fs, old_framerate, window=3.7)
        new_audio_data = np.zeros(audio_data.shape)
        new_audio_data[::2] = new_l
        new_audio_data[1::2] = new_r
    elif ch > 2:
        raise ValueError("Cannot handle more than 2 channel audio")
    else:
        new_audio_data = signal.resample_poly(audio_data, new_fs, old_framerate, window=3.7)

    new_len = len(new_audio_data)
    new_audio_data = new_audio_data.astype(np.int16).tobytes()
    f.close()

    f = wave.open(os.path.join(root, fname), "wb")
    f.setparams((channels, new_samp_width, new_fs, new_len, "NONE", "not compressed"))
    f.writeframes(new_audio_data)
    f.close()

"""
shutil.copytree(kaggle_root, kaggle_dump, dirs_exist_ok=True)
resample_and_resize(kaggle_dump)
for dir, dump_dir in zip(vox_roots, vox_dumps):
    shutil.copytree(dir, dump_dir, dirs_exist_ok=True)
    resample_and_resize(dump_dir)

shutil.copytree(nick_root, nick_dump, dirs_exist_ok=True)
resample_and_resize(nick_dump)
