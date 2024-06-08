import os
from audio_utils import normalize_audio
import torchaudio
import soundfile as sf

out_folder = "./synth_output-matcha-hifigan"
files = [file for file in os.listdir(out_folder) if file.endswith(".wav")]

count = 0
for file in files:
    try:
        yhat, sr = torchaudio.load(f"{out_folder}/{file}")
        yhat_normalized = normalize_audio(yhat, sample_rate=sr)
        sf.write(f'{out_folder}/normalized/{file}', yhat_normalized.numpy(), sr, subtype='PCM_24')
    except Exception as err:
        print(f"failed to normalize {file}: {err}")
        count += 1
        
print(f"total failures: {count}")
