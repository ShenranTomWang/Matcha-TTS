import os
from audio_utils import normalize_audio
import torchaudio
import soundfile as sf
import numpy as np

out_folder = "./synth_output-maliseet-matcha-hifigan"
files = [file for file in os.listdir(out_folder) if file.endswith(".wav")]

count = 0
for file in files:
    try:
        yhat, sr = torchaudio.load(f"{out_folder}/{file}")
        yhat_normalized = normalize_audio(yhat, sample_rate=sr)
        yhat_normalized = yhat_normalized.numpy()
        yhat_normalized = yhat_normalized.transpose()
        sf.write(f'{out_folder}/normalized/{file}', yhat_normalized, sr, subtype='PCM_24')
    except Exception as err:
        print(f"failed to normalize {file}: {err}")
        count += 1

print(f"total failures: {count}")
