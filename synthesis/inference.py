import datetime as dt
import torch

from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse

import synthesis.utils as utils

@torch.inference_mode()
def process_text(text: str, device: torch.DeviceObjType):
    x = torch.tensor(intersperse(text_to_sequence(text, ['ojibwe_cleaners']), 0),dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }

@torch.inference_mode()
def synthesise(
    text: dict,
    model, 
    n_timesteps: int = 10, 
    temperature: float = 0.667, 
    length_scale: float = 1.0, 
    spks: torch.Tensor = None, 
    lang: torch.Tensor = None
):
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
def batch_synthesis(
    texts: list,
    names: list, 
    model, vocoder, denoiser, 
    batch_size: int, 
    hop_length: int, 
    device: torch.DeviceObjType, 
    sr: int, 
    spks: list = None, 
    lang: list = None,
    temperature: float = 0.667,
    length_scale: float = 1.0,
    n_timesteps: int = 10
):
    outputs = []

    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        batch_texts = texts[i:end_idx]
        batch_names = names[i: end_idx]
        batch_spks = torch.tensor(spks[i:end_idx], device=device) if spks is not None else None
        batch_lang = torch.tensor(lang[i:end_idx], device=device) if lang is not None else None

        batch_x = [process_text(text, device) for text in batch_texts]
        batch_lengths = torch.tensor([x["x"].shape[1] for x in batch_x], dtype=torch.long, device=device)
        max_len = int(max(batch_lengths))
        batch_x = torch.cat([utils.pad(process_text(text, device)['x'], max_len) for text in batch_texts], dim=0)
        inputs = {"x": batch_x, "x_lengths": batch_lengths}

        batch_output = synthesise(
            inputs, 
            model, 
            spks=batch_spks, 
            lang=batch_lang,
            temperature=temperature,
            length_scale=length_scale,
            n_timesteps=n_timesteps
        )

        batch_output['waveform'] = to_waveform(batch_output['mel'], denoiser, vocoder)
        rtf_w = utils.compute_rtf_w(batch_output, sr)
        batch_output["waveform_lengths"] = utils.compute_waveform_lengths(batch_output, hop_length)
        batch_output['inference_time'] = utils.compute_time_spent(batch_output)
        batch_output['throughput'] = utils.compute_throughput(batch_output, sr)
        batch_output["waveform"] = utils.trim_waveform(batch_output)
        batch_output["rtf_w"] = rtf_w
        batch_output["names"] = batch_names
        
        utils.batch_report(batch_output, i / batch_size + 1)
        outputs.append(batch_output)
    return outputs


@torch.inference_mode()
def to_waveform(mel, denoiser, vocoder):
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser != None:
        audio = denoiser(audio.squeeze(0), strength=0.00025).cpu()
    return audio.cpu()