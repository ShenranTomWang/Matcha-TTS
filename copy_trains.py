import os
import shutil

# Read the list of files from the .txt file
with open("/project/6080355/shenranw/Matcha-TTS/data/filelists/objiwe_audio_text_train_filelist.txt", "r") as file:
    train_files_list = file.readlines()

with open("/project/6080355/shenranw/Matcha-TTS/data/filelists/objiwe_audio_text_val_filelist.txt", "r") as file:
    val_files_list = file.readlines()

# Define the destination directory
train_destination_dir = "/project/6080355/shenranw/train_samples"
val_destination_dir = "/project/6080355/shenranw/val_samples"

# Iterate over each line in the list of files
count = 0
for line in train_files_list:
    directory, description = line.strip().split("|")  # Split directory path and description
    directory = "/project/6080355/shenranw/Matcha-TTS/" + directory
    destination_file = os.path.join(train_destination_dir, f"{count}.wav")  # Create destination file path
    shutil.copyfile(directory, destination_file)  # Copy file to destination directory
    count += 1

count = 0
for line in val_files_list:
    directory, description = line.strip().split("|")  # Split directory path and description
    directory = "/project/6080355/shenranw/Matcha-TTS/" + directory
    destination_file = os.path.join(val_destination_dir, f"{count}.wav")  # Create destination file path
    shutil.copyfile(directory, destination_file)  # Copy file to destination directory
    count += 1

print("Files copied successfully!")