import torch
import datetime as dt

from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.hifigan.config import v1

from vocos import Vocos

from matcha.models.matcha_tts import MatchaTTS


def pad(input: torch.Tensor, target_len: int):
    padding_needed = target_len - input.size(-1)
    if padding_needed <= 0:
        return input

    return torch.nn.functional.pad(input, (0, padding_needed), "constant", 0)

def trim_waveform(output: dict):
    waveforms = output["waveform"]
    lengths = output["waveform_lengths"]
    trimmed_wavs = []
    for i in range(waveforms.shape[0]):
        waveform = waveforms[i, :]
        length = int(lengths[i])
        trimmed = waveform[:length]
        trimmed = trimmed.unsqueeze(0)
        trimmed_wavs.append(trimmed)
    return trimmed_wavs

def get_item(data: list, device: torch.DeviceObjType):
    path = data[0]
    spks = torch.tensor([int(data[1])], device=device) if data[1] != None else None
    lang = torch.tensor([int(data[2])], device=device) if data[2] != None else None
    text = data[3]
    return path, spks, lang, text

def get_item_batched(ckpt: dict, texts: list, spk_emb: bool, lang_emb: bool):
    hop_length = ckpt["datamodule_hyper_parameters"]["hop_length"]
    paths = [data[0] for data in texts]
    dirs = [path.split("/") for path in paths]
    names = [dir[len(dir) - 1].split(".")[0] for dir in dirs]
    inputs = [data[3] for data in texts]
    spks = [int(data[1]) for data in texts] if spk_emb else None
    lang = [int(data[2]) for data in texts] if lang_emb else None
    return hop_length, names, inputs, spks, lang

def get_data_index(spk_emb: bool, lang_emb: bool):
    if not spk_emb and not lang_emb:
        return 1
    elif spk_emb and not lang_emb:
        return 2
    elif lang_emb and not spk_emb:
        return 2
    else:
        return 3

def pretty_print(output: dict, rtf_w: float, i):
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
    
def batch_report(output: dict, i: int):
    print(f"Batch {i}")
    print(f"Inference time: {output['inference_time']}")
    print(f"Throughput: {output['throughput']}")
    print(f"{'-' * 53}")
    
def compute_rtf_w(output: dict, sr: int):
    t = compute_time_spent(output)
    rtf_w = t * sr / (output['waveform'].shape[-1])
    return rtf_w

def compute_throughput(output: dict, sr: int):
    dur = output['waveform_lengths']
    total_dur = sum(dur) / sr
    inference_time = output['inference_time']
    throughput = total_dur / inference_time
    return throughput.cpu()

def compute_time_spent(output: dict):
    t = (dt.datetime.now() - output['start_t']).total_seconds()
    return t

def compute_waveform_lengths(output: dict, hop_length: int):
    wave_lengths = output["mel_lengths"] * hop_length
    return wave_lengths

def load_vocoder(config_path: str, checkpoint_path: str, device: torch.DeviceObjType, vocoder_type: str = "HiFiGAN"):
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

def load_model(checkpoint_path: str, device: torch.DeviceObjType):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model
