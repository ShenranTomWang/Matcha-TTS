import datetime as dt
from pathlib import Path

import IPython.display as ipd
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

# Hifigan imports
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse
# Vocos imports
from vocos import Vocos

MATCHA_CHECKPOINT = "./logs/train/objiwe/runs/mikmaw/checkpoints/checkpoint_epoch=5299.ckpt"
HIFIGAN_CHECKPOINT = "./matcha/hifigan/g_02500000"
VOCOS_CHECKPOINT = "./logs/vocos/version_20/checkpoints/last.ckpt"
OUTPUT_FOLDER = "synth_output-mikmaw-matcha-vocos"
TEXTS_DIR = "./data/filelists/mikmaw_test_filelist.txt"
VOCOS_CONFIG = "./configs/vocos/vocos-matcha.yaml"
## Number of ODE Solver steps
n_timesteps = 10
## Changes to the speaking rate
length_scale=1.0
## Sampling temperature
temperature = 0.667

def synthesis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(checkpoint_path):
        model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
        model.eval()
        return model
    count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"


    model = load_model(MATCHA_CHECKPOINT)
    print(f"Model loaded! Parameter count: {count_params(model)}")

    def load_vocoder(config_path, checkpoint_path, model="hifigan"):
        if model == "hifigan":
            h = AttrDict(v1)
            hifigan = HiFiGAN(h).to(device)
            hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
            _ = hifigan.eval()
            hifigan.remove_weight_norm()
            return hifigan
        elif model == "vocos":
            vocoder = Vocos.from_hparams(config_path).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint["state_dict"]
            vocoder.load_state_dict(state_dict, strict=False)
            return vocoder

    vocoder = load_vocoder(VOCOS_CONFIG, VOCOS_CHECKPOINT, model="vocos")
    # denoiser = Denoiser(vocoder, mode='zeros')

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
        # audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
        return audio.cpu().squeeze()
        
    def save_to_folder(filename: str, output: dict, folder: str):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        np.save(folder / f'{filename}', output['mel'].cpu().numpy())
        sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')
        
    def save_to_folder(filename: str, output: dict, folder: str):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        np.save(folder / f'{filename}', output['mel'].cpu().numpy())
        sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')

    def parse_filelist_get_text(filelist_path, split_char="|", get_index=1):
        filepaths_and_text = []
        with open(filelist_path, encoding="utf-8") as f:
            for line in f:
                path = line.strip().split(split_char)[0]
                sentence = line.strip().split(split_char)[get_index]
                filepaths_and_text.append([path, sentence])
        return filepaths_and_text

    texts = parse_filelist_get_text(TEXTS_DIR)

    outputs, rtfs = [], []
    rtfs_w = []
    for i, data in enumerate(tqdm(texts)):
        path = data[0]
        text = data[1]
        dirs = path.split("/")
        name = dirs[len(dirs) - 1].split(".")[0]
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
        save_to_folder(name, output, OUTPUT_FOLDER)

    print(f"Number of ODE steps: {n_timesteps}")
    print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
    print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")
    

if __name__ == "__main__":
    synthesis()