import numpy as np
import torch
from tqdm.auto import tqdm

# Hifigan imports
from matcha.hifigan.denoiser import Denoiser
# Evaluation imports
import evaluation
# Normalization imports
from audio_utils import normalize_audio

import synthesis.utils as utils
import synthesis.io as io
import synthesis.inference as inference

import os

VOCODER = "Vocos"
BATCHED_SYNTHESIS = bool(os.getenv("BATCHED_SYNTHESIS"))
BATCH_SIZE = 32

WANDB_PROJECT = f"TTS"
WANDB_NAME = os.getenv("WANDB_NAME") + " Batched" if BATCHED_SYNTHESIS else os.getenv("WANDB_NAME")
WANDB_DATASET = "multilingual-test"
WANDB_ARCH = f"MatchaTTS: language embedding, {VOCODER}: vanilla"

Y_FILELIST = "./data/filelists/multilingual_test_filelist.txt"
OUTPUT_FOLDER = f"synth_output-{WANDB_NAME}"
TEXTS_DIR = "./data/filelists/multilingual_test_filelist.txt"
SYNC_SAVE_DIR = "./"

MATCHA_CHECKPOINT = os.getenv("MATCHA_CHECKPOINT")
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

def synthesis():
    count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"

    model = utils.load_model(MATCHA_CHECKPOINT, device)
    print(f"Model loaded! Parameter count: {count_params(model)}")
    
    if VOCODER == "HiFiGAN":
        vocoder = utils.load_vocoder(None, HIFIGAN_CHECKPOINT, device, vocoder_type=VOCODER)
        denoiser = Denoiser(vocoder, mode='zeros')
    else:
        vocoder = utils.load_vocoder(VOCOS_CONFIG, VOCOS_CHECKPOINT, device, vocoder_type=VOCODER)
        denoiser = None
    index = utils.get_data_index(SPK_EMB, LANG_EMB)
    texts = io.parse_filelist_get_text(TEXTS_DIR, SPK_EMB, LANG_EMB, sentence_index=index)

    outputs, rtfs = [], []
    rtfs_w = []
    metrics = {}
    throughputs = []
    if BATCHED_SYNTHESIS:
        ckpt = torch.load(MATCHA_CHECKPOINT)
        hop_length = ckpt["datamodule_hyper_parameters"]["hop_length"]
        paths = [data[0] for data in texts]
        dirs = [path.split("/") for path in paths]
        names = [dir[len(dir) - 1].split(".")[0] for dir in dirs]
        inputs = [data[3] for data in texts]
        spks = [int(data[1]) for data in texts] if SPK_EMB else None
        lang = [int(data[2]) for data in texts] if LANG_EMB else None
        outputs = inference.batch_synthesis(
            inputs, 
            names, 
            model, 
            vocoder, 
            denoiser, 
            BATCH_SIZE, 
            hop_length, 
            device, 
            SAMPLE_RATE, 
            spks=spks, 
            lang=lang,
            temperature=temperature,
            n_timesteps=n_timesteps,
            length_scale=length_scale
        )
        
        
        for i, output in enumerate(outputs):
            normalized_waveforms = []
            rtf_w = output["rtf_w"]
            rtf_w = rtf_w / BATCH_SIZE
            rtf = output["rtf"] / BATCH_SIZE
            throughput = output['throughput']
            for j, wave in enumerate(output['waveform']):
                try:
                    normalized = normalize_audio(wave, sample_rate=SAMPLE_RATE).t()
                except RuntimeError as err:
                    print(f"{output['names'][j]}: {err}")
                    normalized = wave.t()
                normalized_waveforms.append(normalized)
            
            rtfs.append(rtf)
            rtfs_w.append(rtf_w)
            throughputs.append(throughput)

            output['normalized_waveforms'] = normalized_waveforms
            
            io.save_to_folder_batch(output, OUTPUT_FOLDER, SAMPLE_RATE)
            
    else:
        for i, data in enumerate(tqdm(texts)):
            path = data[0]
            text, spks, lang = utils.get_item(data, SPK_EMB, LANG_EMB, device)
            dirs = path.split("/")
            name = dirs[len(dirs) - 1].split(".")[0]
            output = inference.synthesise(
                inference.process_text(text, device), 
                model, 
                spks=spks, 
                lang=lang,
                temperature=temperature,
                length_scale=length_scale,
                n_timesteps=n_timesteps
            )
            waveform = inference.to_waveform(output['mel'], denoiser, vocoder)
            output['waveform'] = normalize_audio(waveform, sample_rate=SAMPLE_RATE).t().squeeze()
            
            rtf_w = utils.compute_rtf_w(output, SAMPLE_RATE)
            rtfs.append(output['rtf'])
            rtfs_w.append(rtf_w)
            
            utils.pretty_print(output, rtf_w, name)
            io.save_to_folder(name, output, OUTPUT_FOLDER, SAMPLE_RATE)
    
    print(f"Experiment: {WANDB_NAME}")
    
    rtfs = rtfs[1:]
    rtfs_w = rtfs_w[1:]
    throughputs = throughputs[1:]
    
    rtfs_mean = np.mean(rtfs)
    rtfs_std = np.std(rtfs)
    rtfs_w_mean = np.mean(rtfs_w)
    rtfs_w_std = np.std(rtfs_w)
    throughput_mean = np.mean(throughputs)
    throughput_std = np.std(throughputs)
    
    metrics["num_ode_steps"] = n_timesteps
    metrics["rtfs_mean"] = rtfs_mean
    metrics["rtfs_std"] = rtfs_std
    metrics["rtfs_w_mean"] = rtfs_w_mean
    metrics["rtfs_w_std"] = rtfs_w_std
    if BATCHED_SYNTHESIS:
        metrics["throughput_mean"] = throughput_mean
        metrics["throughput_std"] = throughput_std
    
    print(f'"num_ode_steps": {n_timesteps}, "rtfs_mean": {rtfs_mean}, "rtfs_std": {rtfs_std}, "rtfs_w_mean": {rtfs_w_mean}, "rtfs_w_std": {rtfs_w_std}, "throughput_mean": {throughput_mean}, "thoughput_std": {throughput_std}')

    for spk_flag in SPK_FLAGS:
        stoi, pesq, mcd, f0_rmse, las_rmse, vuv_f1, fd = evaluation.evaluate(OUTPUT_FOLDER, Y_FILELIST, spk_flag=spk_flag)
        
        metrics[f"{spk_flag}/stoi"] = stoi
        metrics[f"{spk_flag}/pesq"] = pesq
        metrics[f"{spk_flag}/mcd"] = mcd
        metrics[f"{spk_flag}/f0_rmse"] = f0_rmse
        metrics[f"{spk_flag}/las_rmse"] = las_rmse
        metrics[f"{spk_flag}/vuv_f1"] = vuv_f1
        metrics[f"{spk_flag}/fd"] = fd
        
        print(f'"{spk_flag}/stoi": {stoi}, "{spk_flag}/pesq": {pesq}, "{spk_flag}/mcd": {mcd}, "{spk_flag}/f0_rmse": {f0_rmse}, "{spk_flag}/las_rmse": {las_rmse}, "{spk_flag}/vuv_f1": {vuv_f1}, {spk_flag}/fd": {fd},')
    
    io.save_python_script_with_data(metrics, WANDB_PROJECT, WANDB_NAME, WANDB_ARCH, WANDB_DATASET, device, filename=SYNC_SAVE_DIR + WANDB_NAME.replace(" ", "_") + ".py")

if __name__ == "__main__":
    synthesis()