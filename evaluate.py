from pystoi import stoi
from pesq import pesq
import torchaudio
import torch
from pymcd.mcd import Calculate_MCD
import librosa
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import os

YHAT_FOLDER = "./synth_output-multilingual-matcha-hifigan"
Y_FILELIST = "./data/filelists/multilingual_test_filelist.txt"
SAMPLE_RATE = 22050
FS = 16000
N_FFT = 1024
HOP_LENGTH = 256

device = "cuda" if torch.cuda.is_available() else "cpu"

def pad_shorter_tensor(audio1: torch.Tensor, audio2: torch.Tensor) -> tuple:
    """pad the shorter audio input to length of the longer one

    Args:
        audio1 (torch.Tensor): shape: (num_samples, 1)
        audio2 (torch.Tensor): shape: (num_samples, 1)

    Returns:
        (torch.Tensor, torch.Tensor): padded audios
    """
    padding_size = abs(audio1.shape[1] - audio2.shape[1])
    if padding_size > 0:
        padding = torch.zeros((1, padding_size), device=device, dtype=audio1.dtype)
        if audio1.shape[1] < audio2.shape[1]:
            audio1 = torch.cat((audio1, padding), axis=1)
        else:
            audio2 = torch.cat((audio2, padding), axis=1)
    return audio1, audio2

def pad_shorter_np(audio1: np.ndarray, audio2: np.ndarray) -> tuple:
    """pad the shorter audio input to length of the longer one

    Args:
        audio1 (np.ndarray): shape: (num_samples,)
        audio2 (np.ndarray): shape: (num_samples,)

    Returns:
        (np.ndarray, np.ndarray): padded audios
    """
    padding_size = abs(audio1.shape[0] - audio2.shape[0])
    if padding_size > 0:
        padding = np.zeros(shape=(padding_size,))
        if audio1.shape[0] < audio2.shape[0]:
            audio1 = np.concatenate((audio1, padding))
        else:
            audio2 = np.concatenate((audio2, padding))
    return audio1, audio2

def calculate_f0(audio: np.ndarray) -> np.ndarray:
    """
    Calculate the fundamental frequency (F0) using the librosa library.

    Args:
        audio (np.ndarray): The input audio signal.

    Returns:
        np.ndarray: The F0 values for each frame.
    """
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return np.nan_to_num(f0)

def rmse(ref, syn) -> float:
    """compute RMSE

    Args:
        ref (np.ndarray)
        syn (np.ndarray)

    Returns:
        float: RMSE
    """
    return np.sqrt(np.mean((ref - syn) ** 2))

def f0_rmse(reference_wav: str, synthesized_wav: str, sr=SAMPLE_RATE) -> float:
    """compute F0-RMSE

    Args:
        reference_wav (str): reference .wav filepath
        synthesized_wav (str): synthesized .wav filepath
        sr (int, optional): sample rate. Defaults to SAMPLE_RATE.

    Returns:
        float: F0-RMSE
    """
    ref_audio, _ = librosa.load(reference_wav, sr=sr)
    syn_audio, _ = librosa.load(synthesized_wav, sr=sr)
    
    ref_audio, syn_audio = pad_shorter_np(ref_audio, syn_audio)
    
    f0_ref = calculate_f0(ref_audio)
    f0_syn = calculate_f0(syn_audio)
    
    return rmse(f0_ref, f0_syn)

def las_rmse(reference_wav: str, synthesized_wav: str, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH) -> float:
    """compute LAS-RMSE

    Args:
        reference_wav (str): reference .wav filepath
        synthesized_wav (str): synthesized .wav filepath
        sr (int, optional): sample rate. Defaults to SAMPLE_RATE.
        n_fft (int, optional): n_fft used for STFT. Defaults to N_FFT.
        hop_length (int, optional): hop_length used for STFT. Defaults to HOP_LENGTH.

    Returns:
        float: LAS-RMSE
    """
    ref_audio, _ = librosa.load(reference_wav, sr=sr)
    syn_audio, _ = librosa.load(synthesized_wav, sr=sr)
    
    ref_audio, syn_audio = pad_shorter_np(ref_audio, syn_audio)
    
    ref_stft = librosa.stft(ref_audio, n_fft=n_fft, hop_length=hop_length)
    syn_stft = librosa.stft(syn_audio, n_fft=n_fft, hop_length=hop_length)
    
    ref_amplitude = np.abs(ref_stft)
    syn_amplitude = np.abs(syn_stft)
    
    epsilon = 1e-10     # Convert to log scale (add a small constant to avoid log(0))
    ref_log_amplitude = np.log(ref_amplitude + epsilon)
    syn_log_amplitude = np.log(syn_amplitude + epsilon)
    
    return rmse(ref_log_amplitude, syn_log_amplitude)

def vuv_f1(reference_wav: str, synthesized_wav: str, sr=SAMPLE_RATE) -> tuple:
    """
    Calculate the Voiced/Unvoiced F1 score between reference and synthesized audio signals.

    Args:
        reference_wav (str): Path to the reference audio file.
        synthesized_wav (str): Path to the synthesized audio file.
        sr (int): The sampling rate of the audio signals.

    Returns:
        (precision, recall, f1): Precision, recall and F1 score for the voiced/unvoiced decisions.
    """
    def voiced_unvoiced_decision(f0):
        """
        Determine voiced/unvoiced decisions based on F0 values.

        Args:
            f0 (np.ndarray): The F0 values for each frame.

        Returns:
            np.ndarray: Boolean array where True indicates a voiced frame and False indicates an unvoiced frame.
        """
        return f0 > 0
    ref_audio, _ = librosa.load(reference_wav, sr=sr)
    syn_audio, _ = librosa.load(synthesized_wav, sr=sr)
    
    ref_audio, syn_audio = pad_shorter_np(ref_audio, syn_audio)

    f0_ref = calculate_f0(ref_audio)
    f0_syn = calculate_f0(syn_audio)

    vuv_ref = voiced_unvoiced_decision(f0_ref)
    vuv_syn = voiced_unvoiced_decision(f0_syn)

    _, _, f1, _ = precision_recall_fscore_support(vuv_ref, vuv_syn, average='binary')

    return f1

def main():
    paired_data = {}
    for file in os.listdir(YHAT_FOLDER):
        if file.endswith(".wav"):
            paired_data[file] = {
                "yhat": f"{YHAT_FOLDER}/{file}",
                "y": None
            }
    with open(Y_FILELIST, "r") as fl:
        for line in fl:
            path = line.split("|")[0]
            dirs = path.split("/")
            filename = dirs[len(dirs) - 1]
            paired_data[filename]["y"] = path

    mcd = Calculate_MCD(MCD_mode="plain")

    stoi_score_sum = 0
    pesq_score_sum = 0
    mcd_score_sum = 0
    f0_rmse_sum = 0
    las_rmse_sum = 0
    vuv_f1_sum = 0
    for name in paired_data.keys():
        dirs = paired_data[name]
        y_path, yhat_path = dirs["yhat"], dirs["y"]
        y, _ = torchaudio.load(y_path)
        yhat, _ = torchaudio.load(yhat_path)
        y, yhat = pad_shorter_tensor(y, yhat)
        
        y = torchaudio.functional.resample(y, orig_freq=SAMPLE_RATE, new_freq=FS)
        yhat = torchaudio.functional.resample(yhat, orig_freq=SAMPLE_RATE, new_freq=FS)
        
        f0_rmse_sum += f0_rmse(y_path, yhat_path)
        las_rmse_sum += las_rmse(y_path, yhat_path)
        vuv_f1_sum += vuv_f1(y_path, yhat_path)
        mcd_score_sum += mcd.calculate_mcd(y_path, yhat_path)
        stoi_score_sum += stoi(y.t(), yhat.t(), FS, extended=False)
        pesq_score_sum += pesq(FS, y.squeeze(0).cpu().numpy(), yhat.squeeze(0).cpu().numpy(), "wb")

    size = len(paired_data)
    stoi_mean = stoi_score_sum / size
    pesq_mean = pesq_score_sum / size
    mcd_mean = mcd_score_sum / size
    f0_rmse_mean = f0_rmse_sum / size
    las_rmse_mean = las_rmse_sum / size
    vuv_f1_mean = vuv_f1_sum / size

    print(f"stoi: {stoi_mean}, pesq: {pesq_mean}, mcd: {mcd_mean}, f0-rmse: {f0_rmse_mean}, las-rmse: {las_rmse_mean}, vuv-f1: {vuv_f1_mean}")

if __name__ == "__main__":
    main()
