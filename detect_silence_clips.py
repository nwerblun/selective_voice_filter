import numpy as np
import os
import wave
root = r"C:\Users\NWerblun\Desktop\selective_voice_filter\data\voice_data"
import pdb
silent_clips = []
_, subdirs, _ = next(os.walk(root))
for s in subdirs:
    _,_,fnames = next(os.walk(os.path.join(root, s)))
    for file in fnames:
        f = wave.open(os.path.join(root,s, file), "rb")
        audio_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)
        max_possible = (2**(8*f.getsampwidth()))/2
        f.close()
        rms = np.sqrt(np.mean(audio_data**2))
        dBFS = 10*np.log10(rms/max_possible)
        if dBFS < -19:
            silent_clips += [(os.path.join(root,s,file), dBFS)]

for f in silent_clips:
    print(f[0], "    ", str(f[1]), "dBFS")
