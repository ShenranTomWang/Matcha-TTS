import os
import shutil

# Read the list of files from the .txt file
with open("./data/filelists/multilingual_test_filelist.txt", "r", encoding="utf-8") as file:
    files_list = file.readlines()

# Define the destination directory
destination_dir = "../synth_output/synth_output-multilingual-matcha-hifigan-balanced-dataset/"

# Iterate over each line in the list of files
for line in files_list:
    directory = line.split("|")[0]  # Split directory path and description
    dirs = directory.split("/")
    name = dirs[len(dirs) - 1].split(".")[0]
    destination_file = os.path.join(destination_dir, f"{name}_real.wav")  # Create destination file path
    shutil.copy(directory, destination_file)  # Copy file to destination directory

print("Files copied successfully!")