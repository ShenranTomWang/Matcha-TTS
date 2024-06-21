import os

folder = "./synth_output-maliseet-matcha-hifigan"

filelist = os.listdir(folder)
for file in filelist:
    if file.endswith(".npy"):
        os.remove(f"{folder}/{file}")
