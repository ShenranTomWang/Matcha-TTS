import datetime as dt
from pathlib import Path

import IPython.display as ipd
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm
import wandb

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
# Evaluation imports
import evaluation
# Normalization imports
from audio_utils import normalize_audio

Y_FILELIST = "./data/filelists/multilingual_test_filelist.txt"
OUTPUT_FOLDER = "synth_output-multilingual-matcha-hifigan"
TEXTS_DIR = "./data/filelists/multilingual_test_filelist.txt"

MATCHA_CHECKPOINT = "./logs/train/multilingual/runs/multilingual/checkpoints/last.ckpt"
HIFIGAN_CHECKPOINT = "./matcha/hifigan/g_02500000"
VOCOS_CHECKPOINT = "./logs/vocos/multilingual/checkpoints/last.ckpt"

VOCOS_CONFIG = "./configs/vocos/vocos-matcha.yaml"

WANDB_PROJECT = "MatchaTTS-HiFiGAN"
WANDB_NAME = "Multilingual Experiment A100"
WANDB_DATASET = "multilingual-test"
WANDB_ARCH = "MatchaTTS: language embedding, HiFiGAN: vanilla, general"

VOCODER = "hifigan"
LANG_EMB = True
SPK_EMB = True
SAMPLE_RATE = 22050
## Number of ODE Solver steps
n_timesteps = 10
## Changes to the speaking rate
length_scale=1.0
## Sampling temperature
temperature = 0.667
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vocoder(config_path, checkpoint_path, vocoder_type="hifigan"):
    if vocoder_type == "hifigan":
        h = AttrDict(v1)
        hifigan = HiFiGAN(h).to(device)
        hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
        _ = hifigan.eval()
        hifigan.remove_weight_norm()
        return hifigan
    elif vocoder_type == "vocos":
        vocoder = Vocos.from_hparams(config_path).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        vocoder.load_state_dict(state_dict, strict=False)
        return vocoder

def load_model(checkpoint_path):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model

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
def synthesise(text, model, spks=None, lang=None):
    text_processed = process_text(text)
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'], 
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        lang=lang,
        length_scale=length_scale
    )
    # merge everything to one dict    
    output.update({'start_t': start_t, **text_processed})
    return output

@torch.inference_mode()
def to_waveform(mel, denoiser, vocoder):
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser != None:
        audio = denoiser(audio.squeeze(0), strength=0.00025).cpu()
    audio = normalize_audio(audio, sample_rate=SAMPLE_RATE)
    audio = audio.t()
    return audio.cpu().squeeze()

def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], SAMPLE_RATE, 'PCM_24')

def parse_filelist_get_text(filelist_path, split_char="|", sentence_index=3, spk_index=1, lang_index=2):
    filepaths_and_text = []
    with open(filelist_path, encoding="utf-8") as f:
        for line in f:
            path = line.strip().split(split_char)[0]
            spk = line.strip().split(split_char)[spk_index] if SPK_EMB else None
            lang = line.strip().split(split_char)[lang_index] if LANG_EMB else None
            sentence = line.strip().split(split_char)[sentence_index]
            filepaths_and_text.append([path, spk, lang, sentence])
    return filepaths_and_text

def synthesis():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_NAME,
        config={
            "architecture": WANDB_ARCH,
            "dataset": WANDB_DATASET,
            "hardware": device
        }
    )
    
    count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"

    model = load_model(MATCHA_CHECKPOINT)
    print(f"Model loaded! Parameter count: {count_params(model)}")
    
    if VOCODER == "hifigan":
        vocoder = load_vocoder(None, HIFIGAN_CHECKPOINT, vocoder_type=VOCODER)
        denoiser = Denoiser(vocoder, mode='zeros')
    else:
        vocoder = load_vocoder(VOCOS_CONFIG, VOCOS_CHECKPOINT, vocoder_type=VOCODER)
        denoiser = None

    texts = parse_filelist_get_text(TEXTS_DIR)

    outputs, rtfs = [], []
    rtfs_w = []
    for i, data in enumerate(tqdm(texts)):
        path = data[0]
        if not SPK_EMB and not LANG_EMB:
            text = data[1]
            spks = None
            lang = None
        elif SPK_EMB and not LANG_EMB:
            spks = torch.tensor([int(data[1])], device=device)
            text = data[2]
            lang = None
        elif LANG_EMB and not SPK_EMB:
            lang = torch.tensor([int(data[1])], device=device)
            text = data[2]
            spks = None
        else:
            spks = torch.tensor([int(data[1])], device=device)
            lang = torch.tensor([int(data[2])], device=device)
            text = data[3]
        dirs = path.split("/")
        name = dirs[len(dirs) - 1].split(".")[0]
        output = synthesise(text, model, spks=spks, lang=lang) #, torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
        output['waveform'] = to_waveform(output['mel'], denoiser, vocoder)

        # Compute Real Time Factor (RTF) with HiFi-GAN
        t = (dt.datetime.now() - output['start_t']).total_seconds()
        rtf_w = t * SAMPLE_RATE / (output['waveform'].shape[-1])

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
        ipd.display(ipd.Audio(output['waveform'], rate=SAMPLE_RATE))

        ## Save the generated waveform
        save_to_folder(name, output, OUTPUT_FOLDER)

    stoi, pesq, mcd, f0_rmse, las_rmse, vuv_f1 = evaluation.evaluate(OUTPUT_FOLDER, Y_FILELIST)
    rtfs_mean = np.mean(rtfs)
    rtfs_std = np.std(rtfs)
    rtfs_w_mean = np.mean(rtfs_w)
    rtfs_w_std = np.std(rtfs_w)
    
    wandb.log(
        {
            "stoi": stoi,
            "pesq": pesq,
            "mcd": mcd,
            "f0_rmse": f0_rmse,
            "las_rmse": las_rmse,
            "vuv_f1": vuv_f1,
            "rtfs_mean": rtfs_mean,
            "rtfs_std": rtfs_std,
            "rtfs_w_mean": rtfs_w_mean,
            "rtfs_w_std": rtfs_w_std,
            "num_ode_steps": n_timesteps
        }
    )
    print(f"stoi: {stoi}, pesq: {pesq}, mcd: {mcd}, f0_rmse: {f0_rmse}, las_rmse: {las_rmse}, vuv_f1: {vuv_f1}")
    print(f"Number of ODE steps: {n_timesteps}")
    print(f"Mean RTF:\t\t\t\t{rtfs_mean:.6f} ± {rtfs_std:.6f}")
    print(f"Mean RTF Waveform (incl. vocoder):\t{rtfs_w_mean:.6f} ± {rtfs_w_std:.6f}")

if __name__ == "__main__":
    synthesis()