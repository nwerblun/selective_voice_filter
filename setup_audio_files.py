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
kaggle_dump = "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data"
vox_roots = [
            "C:\\Users\\NWerblun\\Downloads\\mojomove411-20071206\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\chocoholic-20070524\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\chocoholic-20070523\\wav", \
            "C:\\Users\\NWerblun\\Downloads\\calamity-20071011-poe\\wav", \
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
zhe_root = "C:\\Users\\NWerblun\\Downloads\\zhe"
zhe_dump = "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data\\zhe_dump"
nick_test_root = "C:\\Users\\NWerblun\\Downloads\\nick_test"
nick_test_dump = "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\test_data\\nick_test_dump"
silence_test_root = "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\voice_data"
silence_dump = "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\noise_data\\silence"
white_noise_dump = "C:\\Users\\NWerblun\\Desktop\\selective_voice_filter\\data\\noise_data\\white_noise"

FILE_LEN = 1

def resample_and_resize(root, new_len=1):
    _, subdirs, filenames = next(os.walk(root))
    for f in filenames:
        if os.path.splitext(f)[1] == ".wav":
            resample_and_split(root, f, 44100, new_len)
    for s in subdirs:
        resample_and_resize(os.path.join(root, s), new_len)

def resample_and_split(root, fname, new_fs, file_len):
    #File len is desired length in seconds. It will be split up if longer.
    _resample(root, fname, new_fs)
    _split_file(root, fname, file_len)

def move_silent_clips(root, dump_loc):
    _, subdirs, _ = next(os.walk(root))
    for s in subdirs:
        _,_,filenames = next(os.walk(os.path.join(root, s)))
        for f in filenames:
            if os.path.splitext(f)[1] == ".wav":
                if _is_silent(os.path.join(root, s, f)):
                    new_fname = s+"_"+f
                    os.replace(os.path.join(root, s, f), os.path.join(dump_loc, new_fname))

def _split_file(root, fname, file_len):
    f = wave.open(os.path.join(root, fname), "rb")
    fs = f.getframerate()
    nf = f.getnframes()
    if (nf/fs == file_len):
        #print("File length ok, skipping")
        return
    if fs*file_len - int(fs*file_len) > 0:
        print("file length will be changed to {} to match sampling freq.".format(int(fs*file_len)/fs))
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
    if nf % (fs*file_len) > 0:
        print("padding a few zeros to reach a multiple of the fs*file_len")
        to_pad = int(fs*file_len - (nf%(fs*file_len)))
    padded = np.zeros((1, audio_data.shape[1]+to_pad+extra_zeros))
    padded[:,:audio_data.shape[1]] = audio_data
    #Make each col. = file_len seconds of data
    audio_data = np.reshape(padded, (padded.shape[1]//int(fs*file_len), int(fs*file_len)))
    #write each row as a new file
    for ind, row in enumerate(audio_data):
        new_fname = os.path.splitext(fname)[0] + "_" + str(ind) + os.path.splitext(fname)[1]
        f = wave.open(os.path.join(root, new_fname), "wb")
        f.setparams((1, sw, fs, len(row), "NONE", "not compressed"))
        f.writeframes(row.astype(np.int16).tobytes())
        f.close()
    os.remove(os.path.join(root, fname))

def _resample(root, fname, new_fs):
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
    elif channels > 2:
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

def _is_silent(file_path):
    f = wave.open(file_path, "rb")
    audio_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)
    #2^Num_bits is largest ADC val. Half for pos, half for neg, so /2
    max_possible = (2**(8*f.getsampwidth()))/2
    f.close()
    rms = np.sqrt(np.mean(audio_data**2))
    peak = np.max(audio_data)
    dBFS = 10*np.log10(rms/max_possible)
    return dBFS <= -20

def make_white_noise(file_len, fs=44100, samp_width=2, rms=-15):
    #rms is in dBFS
    if samp_width == 2:
        tp = np.int16
        std = (10**(rms/10)) * (2**16)/2
    else:
        tp = np.int8
        std = (10**(rms/10)) * (2**8)/2

    num_samples = int(fs * file_len)
    return np.random.normal(loc=0, scale=std, size=(num_samples,)).astype(tp).tobytes()

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

print("Moving and resampling audio clips...")
shutil.copytree(kaggle_root, kaggle_dump, dirs_exist_ok=True)
resample_and_resize(kaggle_dump, FILE_LEN)

os.makedirs(kaggle_dump+"\\..\\noise_data", exist_ok=True)
shutil.move(kaggle_dump+"\\other", kaggle_dump+"\\..\\noise_data")
shutil.move(kaggle_dump+"\\background_noise", kaggle_dump+"\\..\\noise_data")

for dir, dump_dir in zip(vox_roots, vox_dumps):
    shutil.copytree(dir, dump_dir, dirs_exist_ok=True)
    resample_and_resize(dump_dir, FILE_LEN)

#Make some more samples of my voice by offsetting the audio
make_offset_clip(nick_root+"\\nick.wav", 2123431, 8102324)
make_offset_clip(nick_root+"\\nick2.wav", 1123431, 4102324)
make_offset_clip(nick_root+"\\nick3.wav", 823431, 7102324)
make_offset_clip(nick_root+"\\nick4.wav", 91200, 510324)

shutil.copytree(nick_root, nick_dump, dirs_exist_ok=True)
resample_and_resize(nick_dump, FILE_LEN)

#Corrupted, skip
# shutil.copytree(zhe_root, zhe_dump, dirs_exist_ok=True)
# resample_and_resize(zhe_dump, FILE_LEN)

shutil.copytree(nick_test_root, nick_test_dump, dirs_exist_ok=True)
resample_and_resize(nick_test_dump, FILE_LEN)

# Go through everything again and detect if it's a clip of silence.
print("Moving silent clips to noise directory...")
os.makedirs(silence_dump, exist_ok=True)
move_silent_clips(silence_test_root, silence_dump)

print("Generating white noise...")
# Make a bunch of white noise clips
os.makedirs(white_noise_dump, exist_ok=True)
for i in range(50):
    wn = make_white_noise(FILE_LEN, rms=-10)
    f = wave.open(white_noise_dump+"\\white_noise_"+str(i)+".wav", "wb")
    f.setparams((1, 2, 44100, 44100, "NONE", "not compressed"))
    f.writeframes(wn)
    f.close()

#Make pure silence clips, canceled for now because I don't think it helps.
s = np.zeros((44100,)).astype(np.int16).tobytes()
for i in range(0):
    f = wave.open(silence_dump+"\\silent_"+str(i)+".wav", "wb")
    f.setparams((1, 2, 44100, 44100, "NONE", "not compressed"))
    f.writeframes(s)
    f.close()
