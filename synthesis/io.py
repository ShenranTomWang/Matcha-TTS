import numpy as np
import soundfile as sf
from pathlib import Path


def save_to_folder(filename: str, output: dict, folder: str, sr: int):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], sr, 'PCM_24')
    
def save_to_folder_batch(output: dict, folder: str, sr: int):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    for i in range(output["mel"].shape[0]):
        filename = output["names"][i]
        np.save(folder / f'{filename}', output['mel'][i].cpu().numpy())
        sf.write(folder / f'{filename}.wav', output['normalized_waveforms'][i], sr, 'PCM_24')

def parse_filelist_get_text(
    filelist_path: str, 
    spk_emb: bool, 
    lang_emb: bool, 
    split_char: str = "|", 
    sentence_index: int = 3, 
    spk_index: int = 1, 
    lang_index: int = 2
):
    filepaths_and_text = []
    with open(filelist_path, encoding="utf-8") as f:
        for line in f:
            path = line.strip().split(split_char)[0]
            spk = line.strip().split(split_char)[spk_index] if spk_emb else None
            lang = line.strip().split(split_char)[lang_index] if lang_emb else None
            sentence = line.strip().split(split_char)[sentence_index]
            filepaths_and_text.append([path, spk, lang, sentence])
    return filepaths_and_text

def save_python_script_with_data(
    metrics: dict, 
    project_name: str, 
    run_name: str, 
    arch: str, 
    dataset: str, 
    device: str, 
    filename: str = "sync_wandb.py"
):
    with open(filename, "w") as f:
        f.write(
            f"import wandb\n"
            f"metrics = " + str(metrics) + "\n\n"
            f"def sync_wandb(data, project_name, run_name, config):\n"
            f"    wandb.init(project=project_name, name=run_name, config=config)\n"
            f"    wandb.log(data)\n\n"
            f"if __name__ == '__main__':\n"
            f"    project_name = '{project_name}'\n"
            f"    run_name = '{run_name}'\n"
            "    config = {\n"
            f"        'architecture': '{arch}',\n"
            f"        'dataset': '{dataset}',\n"
            f"        'hardware': '{device}',\n"
            "    }\n"
            f"    sync_wandb(metrics, project_name, run_name, config)\n"
        )