import datetime as dt
from pathlib import Path

import IPython.display as ipd
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

# Hifigan imports
from vocos.vocos.pretrained import Vocos
# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import get_user_data_dir, intersperse
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MATCHA_CHECKPOINT = "/project/6080355/shenranw/Matcha-TTS/logs/train/objiwe/runs/2024-04-16_20-57-00/checkpoints/last.ckpt"
HIFIGAN_CHECKPOINT = "/project/6080355/shenranw/Matcha-TTS/matcha/hifigan/g_02500000"
OUTPUT_FOLDER = "synth_output-vocos"
TEXTS_DIR = "/project/6080355/shenranw/Matcha-TTS/data/filelists/objiwe_audio_text_test_filelist.txt"
VOCOS_CONFIG = "/project/6080355/shenranw/Matcha-TTS/vocos/configs/vocos-matcha.yml"
VOCOS_CHECKPOINT = "/project/6080355/shenranw/Matcha-TTS/vocos/logs/li9ghtning_logs/version_5/checkpoints/last.ckpd"
## Number of ODE Solver steps
n_timesteps = 10
## Changes to the speaking rate
length_scale=1.0
## Sampling temperature
temperature = 0.667

def load_model(checkpoint_path):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model
count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"


model = load_model(MATCHA_CHECKPOINT)
print(f"Model loaded! Parameter count: {count_params(model)}")

def load_vocoder(config_path, checkpoint_path):
    vocos = Vocos.from_checkpoint(config_path, checkpoint_path)
    return vocos

vocoder = load_vocoder(VOCOS_CONFIG, VOCOS_CHECKPOINT)

@torch.inference_mode()
def process_text(text: str):
    x = torch.tensor(intersperse(text_to_sequence(text, ['objiwe_cleaners']), 0),dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }


@torch.inference_mode()
def synthesise(text, spks=None):
    text_processed = process_text(text)
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'], 
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        length_scale=length_scale
    )
    # merge everything to one dict    
    output.update({'start_t': start_t, **text_processed})
    return output

@torch.inference_mode()
def to_waveform(mel, vocoder):
    audio = vocoder(mel).clamp(-1, 1)
    return audio.cpu().squeeze()
    
def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')

def parse_filelist_get_text(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char)[1] for line in f]
    return filepaths_and_text

texts = parse_filelist_get_text(TEXTS_DIR)

outputs, rtfs = [], []
rtfs_w = []
for i, text in enumerate(tqdm(texts)):
    output = synthesise(text) #, torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
    output['waveform'] = to_waveform(output['mel'], vocoder)

    # Compute Real Time Factor (RTF) with HiFi-GAN
    t = (dt.datetime.now() - output['start_t']).total_seconds()
    rtf_w = t * 22050 / (output['waveform'].shape[-1])

    ## Pretty print
    print(f"{'*' * 53}")
    print(f"Input text - {i}")
    print(f"{'-' * 53}")
    print(output['x_orig'])
    print(f"{'*' * 53}")
    print(f"Phonetised text - {i}")
    print(f"{'-' * 53}")
    print(output['x_phones'])
    print(f"{'*' * 53}")
    print(f"RTF:\t\t{output['rtf']:.6f}")
    print(f"RTF Waveform:\t{rtf_w:.6f}")
    rtfs.append(output['rtf'])
    rtfs_w.append(rtf_w)

    ## Display the synthesised waveform
    ipd.display(ipd.Audio(output['waveform'], rate=22050))

    ## Save the generated waveform
    save_to_folder(i, output, OUTPUT_FOLDER)

print(f"Number of ODE steps: {n_timesteps}")
print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")