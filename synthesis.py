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
# Evaluation imports
import evaluation
# Normalization imports
from audio_utils import normalize_audio

VOCODER = "Vocos"
BATCHED_SYNTHESIS = True

WANDB_PROJECT = f"TTS"
WANDB_NAME = f"Multilingual Experiment A100 {VOCODER} Balanced Dataset Mamba2 38M"
WANDB_DATASET = "multilingual-test"
WANDB_ARCH = f"MatchaTTS: language embedding, {VOCODER}: vanilla"

Y_FILELIST = "./data/filelists/multilingual_test_filelist.txt"
OUTPUT_FOLDER = f"synth_output-{WANDB_NAME}"
TEXTS_DIR = "./data/filelists/multilingual_test_filelist.txt"
SYNC_SAVE_DIR = "./"

MATCHA_CHECKPOINT = "./logs/train/multilingual_mamba2/runs/38M/checkpoints/last.ckpt"
HIFIGAN_CHECKPOINT = "./matcha/hifigan/g_02500000"
VOCOS_CHECKPOINT = "./logs/vocos/multilingual-balanced-dataset/checkpoints/last.ckpt"

VOCOS_CONFIG = "./configs/vocos/vocos-matcha.yaml"

LANG_EMB = True
SPK_EMB = True
SPK_FLAGS = ["AT", "MJ", "JJ", "NJ"]
SAMPLE_RATE = 22050
## Number of ODE Solver steps
n_timesteps = 10
## Changes to the speaking rate
length_scale = 1.0
## Sampling temperature
temperature = 0.667
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vocoder(config_path, checkpoint_path, vocoder_type="HiFiGAN"):
    if vocoder_type == "HiFiGAN":
        h = AttrDict(v1)
        hifigan = HiFiGAN(h).to(device)
        hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
        _ = hifigan.eval()
        hifigan.remove_weight_norm()
        return hifigan
    elif vocoder_type == "Vocos":
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
    start_t = dt.datetime.now()
    output = model.synthesise(
        text['x'], 
        text['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        lang=lang,
        length_scale=length_scale
    )
    # merge everything to one dict    
    output.update({'start_t': start_t, **text})
    return output

@torch.inference_mode()
def batch_synthesis(texts, names, model, vocoder, denoiser, batch_size, spks=None, lang=None):
    outputs = []
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        batch_texts = texts[i:end_idx]
        batch_names = names[i: end_idx]
        batch_spks = torch.tensor(spks[i:end_idx], device=device) if spks is not None else None
        batch_lang = torch.tensor(lang[i:end_idx], device=device) if lang is not None else None

        batch_x = [process_text(text) for text in batch_texts]
        batch_lengths = torch.tensor([x["x"].shape[1] for x in batch_x], dtype=torch.long, device=device)
        max_len = int(max(batch_lengths))
        batch_x = torch.cat([pad(process_text(text)['x'], max_len) for text in batch_texts], dim=0)
        inputs = {"x": batch_x, "x_lengths": batch_lengths}

        batch_output = synthesise(inputs, model, batch_spks, batch_lang)

        batch_output['waveform'] = to_waveform(batch_output['mel'], denoiser, vocoder)
        rtf_w = compute_rtf_w(batch_output)
        batch_output["rtf_w"] = rtf_w
        batch_output["names"] = batch_names
        
        outputs.append(batch_output)
        print(f"Samples {i} synthesized")
    print("All samples synthesized")
    return outputs


@torch.inference_mode()
def to_waveform(mel, denoiser, vocoder):
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser != None:
        audio = denoiser(audio.squeeze(0), strength=0.00025).cpu()
    audio = audio.t()
    return audio.cpu().squeeze()

def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], SAMPLE_RATE, 'PCM_24')
    
def save_to_folder_batch(output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    for i in range(output["mel"].shape[0]):
        filename = output["names"][i]
        np.save(folder / f'{filename}', output['mel'][i].cpu().numpy())
        sf.write(folder / f'{filename}.wav', output['waveform'][:, i], SAMPLE_RATE, 'PCM_24')

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

def save_python_script_with_data(metrics, filename="sync_wandb.py"):
    with open(filename, "w") as f:
        f.write(
            f"import wandb\n"
            f"metrics = " + str(metrics) + "\n\n"
            f"def sync_wandb(data, project_name, run_name, config):\n"
            f"    wandb.init(project=project_name, name=run_name, config=config)\n"
            f"    wandb.log(data)\n\n"
            f"if __name__ == '__main__':\n"
            f"    project_name = '{WANDB_PROJECT}'\n"
            f"    run_name = '{WANDB_NAME}'\n"
            "    config = {\n"
            f"        'architecture': '{WANDB_ARCH}',\n"
            f"        'dataset': '{WANDB_DATASET}',\n"
            f"        'hardware': '{device}',\n"
            "    }\n"
            f"    sync_wandb(metrics, project_name, run_name, config)\n"
        )
        
def pad(input, target_len):
    padding_needed = target_len - input.size(-1)
    if padding_needed <= 0:
        return input

    return torch.nn.functional.pad(input, (0, padding_needed), "constant", 0)

def get_item(data):
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
    return text, spks, lang

def get_index():
    if not SPK_EMB and not LANG_EMB:
        return 1
    elif SPK_EMB and not LANG_EMB:
        return 2
    elif LANG_EMB and not SPK_EMB:
        return 2
    else:
        return 3

def pretty_print(output, rtf_w, i):
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
    
def compute_rtf_w(output):
    t = (dt.datetime.now() - output['start_t']).total_seconds()
    rtf_w = t * SAMPLE_RATE / (output['waveform'].shape[-1])
    return rtf_w

def synthesis():
    count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"

    model = load_model(MATCHA_CHECKPOINT)
    print(f"Model loaded! Parameter count: {count_params(model)}")
    
    if VOCODER == "HiFiGAN":
        vocoder = load_vocoder(None, HIFIGAN_CHECKPOINT, vocoder_type=VOCODER)
        denoiser = Denoiser(vocoder, mode='zeros')
    else:
        vocoder = load_vocoder(VOCOS_CONFIG, VOCOS_CHECKPOINT, vocoder_type=VOCODER)
        denoiser = None

    texts = parse_filelist_get_text(TEXTS_DIR)

    outputs, rtfs = [], []
    rtfs_w = []
    metrics = {}
    if BATCHED_SYNTHESIS:
        ckpt = torch.load(MATCHA_CHECKPOINT)
        batch_size = ckpt["datamodule_hyper_parameters"]["batch_size"]
        index = get_index()
        paths = [data[0] for data in texts]
        dirs = [path.split("/") for path in paths]
        names = [dir[len(dir) - 1].split(".")[0] for dir in dirs]
        inputs = [data[index] for data in texts]  # Assuming text is at index 3
        spks = [int(data[1]) for data in texts] if SPK_EMB else None
        lang = [int(data[2]) for data in texts] if LANG_EMB else None
        outputs = batch_synthesis(inputs, names, model, vocoder, denoiser, batch_size, spks=spks, lang=lang)
        
        for i, output in enumerate(outputs):
            rtf_w = output["rtf_w"]
            rtf_w = rtf_w / batch_size
            rtf = output["rtf"] / batch_size
            
            rtfs.append(rtf)
            rtfs_w.append(rtf_w)

            # pretty_print(output, rtf_w, name)
            save_to_folder_batch(output, OUTPUT_FOLDER)
    else:
        for i, data in enumerate(tqdm(texts)):
            path = data[0]
            text, spks, lang = get_item(data)
            dirs = path.split("/")
            name = dirs[len(dirs) - 1].split(".")[0]
            output = synthesise(process_text(text), model, spks=spks, lang=lang) #, torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
            waveform = to_waveform(output['mel'], denoiser, vocoder)
            output['waveform'] = normalize_audio(waveform, sample_rate=SAMPLE_RATE)
            
            rtf_w = compute_rtf_w(output)
            rtfs.append(output['rtf'])
            rtfs_w.append(rtf_w)
            
            pretty_print(output, rtf_w, name)
            save_to_folder(name, output, OUTPUT_FOLDER)
    
    print(f"Experiment: {WANDB_NAME}")
    for spk_flag in SPK_FLAGS:
        stoi, pesq, mcd, f0_rmse, las_rmse, vuv_f1 = evaluation.evaluate(OUTPUT_FOLDER, Y_FILELIST, spk_flag=spk_flag)
        
        metrics[f"{spk_flag}/stoi"] = stoi
        metrics[f"{spk_flag}/pesq"] = pesq
        metrics[f"{spk_flag}/mcd"] = mcd
        metrics[f"{spk_flag}/f0_rmse"] = f0_rmse
        metrics[f"{spk_flag}/las_rmse"] = las_rmse
        metrics[f"{spk_flag}/vuv_f1"] = vuv_f1
        
        print(f'"{spk_flag}/stoi": {stoi}, "{spk_flag}/pesq": {pesq}, "{spk_flag}/mcd": {mcd}, "{spk_flag}/f0_rmse": {f0_rmse}, "{spk_flag}/las_rmse": {las_rmse}, "{spk_flag}/vuv_f1": {vuv_f1}, ')
    
    rtfs_mean = np.mean(rtfs)
    rtfs_std = np.std(rtfs)
    rtfs_w_mean = np.mean(rtfs_w)
    rtfs_w_std = np.std(rtfs_w)
    
    metrics["num_ode_steps"] = n_timesteps
    metrics["rtfs_mean"] = rtfs_mean
    metrics["rtfs_std"] = rtfs_std
    metrics["rtfs_w_mean"] = rtfs_w_mean
    metrics["rtfs_w_std"] = rtfs_w_std
    
    print(f'"num_ode_steps": {n_timesteps}, "rtfs_mean": {rtfs_mean}, "rtfs_std": {rtfs_std}, "rtfs_w_mean": {rtfs_w_mean}, "rtfs_w_std": {rtfs_w_std}')

    save_python_script_with_data(metrics, filename=SYNC_SAVE_DIR + WANDB_NAME.replace(" ", "_") + ".py")

if __name__ == "__main__":
    synthesis()