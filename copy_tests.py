import os
import shutil

# Read the list of files from the .txt file
with open("data/filelists/objiwe_audio_text_test_filelist.txt", "r") as file:
    files_list = file.readlines()

# Define the destination directory
destination_dir = "/project/6080355/shenranw/test_samples"

# Iterate over each line in the list of files
for line in files_list:
    directory, description = line.strip().split("|")  # Split directory path and description
    dirs = directory.split("/")
    name = dirs[len(dirs) - 1].split(".")[0]
    destination_file = os.path.join(destination_dir, f"{name}.wav")  # Create destination file path
    shutil.copyfile(directory, destination_file)  # Copy file to destination directory

print("Files copied successfully!")