from lib2to3.fixes import fix_types
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

"""
Synthesis code.
This script generates a python file that when invoked, syncs all metrics to WandB. Set the run name in environment variable WANDB_NAME
If you want to perform batched synthesis, please set the environment variable BATCHED_SYNTHESIS to 1, and set BATCH_SIZE below
There are two vocoders available: Vocos or HiFiGAN. Please set accordingly below in the VOCODER constant
Y_FILELIST is the path of filelist of all test files
Set the environment variable MATCHA_CHECKPOINT for path of checkpoint where MatchaTTS gets picked up
Set environment variables LANG_EMB or SPK_EMB to 1 if the filelist contains language/speaker embeddings
For Monolingual Synthesis, set environment variable SPK_FLAG_MONOLINGUAL to the corresponding speaker (AT, MJ, JJ, NJ)
"""

VOCODER = "Vocos"
BATCHED_SYNTHESIS = os.getenv("BATCHED_SYNTHESIS") == "1"
BATCH_SIZE = 400

DATA_TYPE = os.getenv("DATA_TYPE")

WANDB_PROJECT = f"TTS"
wandb_name = os.getenv("WANDB_NAME") + " Batched" if BATCHED_SYNTHESIS else os.getenv("WANDB_NAME")
wandb_name = wandb_name + " " + DATA_TYPE if DATA_TYPE != None else wandb_name
WANDB_NAME = wandb_name
WANDB_DATASET = "multilingual-test"
WANDB_ARCH = f"MatchaTTS: language embedding, {VOCODER}: vanilla"

Y_FILELIST = os.getenv("Y_FILELIST")
OUTPUT_FOLDER = f"synth_output-{WANDB_NAME}"
SYNC_SAVE_DIR = "./"

MEM_MAX_ENTRIES = 100000
MEM_FILE_NAME = WANDB_NAME + ".pickle"

MATCHA_CHECKPOINT = os.getenv("MATCHA_CHECKPOINT")
HIFIGAN_CHECKPOINT = "./matcha/hifigan/g_02500000"
VOCOS_CHECKPOINT = "./logs/vocos/multilingual-balanced-dataset/checkpoints/last.ckpt"

VOCOS_CONFIG = "./configs/vocos/vocos-matcha.yaml"

LANG_EMB = os.getenv("LANG_EMB") == "1"
SPK_EMB = os.getenv("SPK_EMB") == "1"
SPK_FLAGS = ["AT", "MJ", "JJ", "NJ"]
SPK_FLAG_MONOLINGUAL = os.getenv("SPK_FLAG_MONOLINGUAL")
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
    texts = io.parse_filelist_get_text(Y_FILELIST, SPK_EMB, LANG_EMB, sentence_index=index)

    outputs, rtfs = [], []
    rtfs_w = []
    metrics = {}
    throughputs = []
    if BATCHED_SYNTHESIS:
        ckpt = torch.load(MATCHA_CHECKPOINT)
        hop_length, names, inputs, spks, lang = utils.get_item_batched(ckpt, texts, SPK_EMB, LANG_EMB)
        for i in range(5):
            print(f"compile run {i}")
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
        print(f"synthesis starting")
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
        # compilation runs
        for i in range(10):
            print(f"compile run {i}")
            data = texts[i]
            path, spks, lang, text = utils.get_item(data, device)
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
            
        print(f"starting synthesis")
        for i, data in enumerate(tqdm(texts)):
            path, spks, lang, text = utils.get_item(data, device)
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
    
    # rtfs = rtfs[1:]
    # rtfs_w = rtfs_w[1:]
    # throughputs = throughputs[1:]
    
    rtfs_mean = np.mean(rtfs)
    rtfs_std = np.std(rtfs)
    rtfs_w_mean = np.mean(rtfs_w)
    rtfs_w_std = np.std(rtfs_w)
    throughput_mean = np.mean(throughputs)
    throughput_std = np.std(throughputs)
    
    metrics["num_ode_steps"] = n_timesteps
    if not BATCHED_SYNTHESIS:
        metrics["rtfs_mean"] = rtfs_mean
        metrics["rtfs_std"] = rtfs_std
        metrics["rtfs_w_mean"] = rtfs_w_mean
        metrics["rtfs_w_std"] = rtfs_w_std
    if BATCHED_SYNTHESIS:
        metrics["throughput_mean"] = throughput_mean
        metrics["throughput_std"] = throughput_std
    
    print(f'"num_ode_steps": {n_timesteps}, "rtfs_mean": {rtfs_mean}, "rtfs_std": {rtfs_std}, "rtfs_w_mean": {rtfs_w_mean}, "rtfs_w_std": {rtfs_w_std}, "throughput_mean": {throughput_mean}, "thoughput_std": {throughput_std}')

    if LANG_EMB:
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
    else:
        stoi, pesq, mcd, f0_rmse, las_rmse, vuv_f1, fd = evaluation.evaluate(OUTPUT_FOLDER, Y_FILELIST, SPK_FLAG_MONOLINGUAL)

        metrics[f"{SPK_FLAG_MONOLINGUAL}/stoi"] = stoi
        metrics[f"{SPK_FLAG_MONOLINGUAL}/pesq"] = pesq
        metrics[f"{SPK_FLAG_MONOLINGUAL}/mcd"] = mcd
        metrics[f"{SPK_FLAG_MONOLINGUAL}/f0_rmse"] = f0_rmse
        metrics[f"{SPK_FLAG_MONOLINGUAL}/las_rmse"] = las_rmse
        metrics[f"{SPK_FLAG_MONOLINGUAL}/vuv_f1"] = vuv_f1
        metrics[f"{SPK_FLAG_MONOLINGUAL}/fd"] = fd
        
        print(f'"{SPK_FLAG_MONOLINGUAL}/stoi": {stoi}, "{SPK_FLAG_MONOLINGUAL}/pesq": {pesq}, "{SPK_FLAG_MONOLINGUAL}/mcd": {mcd}, "{SPK_FLAG_MONOLINGUAL}/f0_rmse": {f0_rmse}, "{SPK_FLAG_MONOLINGUAL}/las_rmse": {las_rmse}, "{SPK_FLAG_MONOLINGUAL}/vuv_f1": {vuv_f1}, {SPK_FLAG_MONOLINGUAL}/fd": {fd},')
    
    io.save_python_script_with_data(metrics, WANDB_PROJECT, WANDB_NAME, WANDB_ARCH, WANDB_DATASET, device, filename=SYNC_SAVE_DIR + WANDB_NAME.replace(" ", "_") + ".py")

if __name__ == "__main__":
    torch.cuda.memory._record_memory_history(max_entries=MEM_MAX_ENTRIES)
    if DATA_TYPE == "fp16":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            synthesis()
    elif DATA_TYPE == "bf16":
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            synthesis()
    else:
        synthesis()
    torch.cuda.memory._dump_snapshot(MEM_FILE_NAME)
    torch.cuda.memory._record_memory_history(enabled=None)
