import datetime as dt
from pathlib import Path

import IPython.display as ipd
import numpy as np
import soundfile as sf
import torch
import json

# Hifigan imports
from matcha.hifigan.config import v1
from matcha.hifigan.meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIFIGAN_CHECKPOINT = "/project/6080355/shenranw/Matcha-TTS/matcha/hifigan/g_02500000"
OUTPUT_FOLDER = "vocoder_test_output"
TEXTS_DIR = "/project/6080355/shenranw/Matcha-TTS/data/filelists/objiwe_audio_text_test_filelist.txt"

# Setup config
h = AttrDict(v1)
torch.manual_seed(h.seed)

def load_vocoder(checkpoint_path):
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

vocoder = load_vocoder(HIFIGAN_CHECKPOINT)
# denoiser = Denoiser(vocoder, mode='zeros')
print("Vocoder loaded")

@torch.inference_mode()
def to_waveform(mel, vocoder):
    audio = vocoder(mel)
    audio = audio.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio
    
def save_to_folder(filename: str, output: dict, folder: str, sr=22050):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    sf.write(folder / f'{filename}.wav', output, sr, 'PCM_24')

def parse_filelist_get_text(filelist_path, split_char="|"):
    filelist, texts = [], []
    with open(filelist_path, encoding="utf-8") as f:
        for line in f:
            file, text = line.strip().split(split_char)
            file = "/project/6080355/shenranw/Matcha-TTS/" + file
            filelist.append(file)
            texts.append(text)
    return filelist, texts
filelist, texts = parse_filelist_get_text(TEXTS_DIR)

for i in range(len(filelist)):
    text = texts[i]
    file = filelist[i]
    print(file)
    wav, sr = load_wav(file)
    print(f"Sample rate: {sr}")
    wav = wav / MAX_WAV_VALUE
    wav = torch.FloatTensor(wav).to(device)
    wav = wav.unsqueeze(0)
    x = mel_spectrogram(wav, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
    output = to_waveform(x, vocoder)

    ## Pretty print
    print(f"{'*' * 53}")
    print(f"Input text - {i}")
    print(f"{'-' * 53}")
    print(text)
    print(f"{'*' * 53}")

    ## Save the generated waveform
    save_to_folder(i, output, OUTPUT_FOLDER, sr=sr)