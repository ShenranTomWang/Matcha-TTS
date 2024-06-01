import os
import shutil

# Read the list of files from the .txt file
with open("data/filelists/objiwe_audio_text_test_filelist.txt", "r") as file:
    files_list = file.readlines()

# Define the destination directory
destination_dir = "/project/6080355/shenranw/test_samples"

# Iterate over each line in the list of files
count = 0
for line in files_list:
    directory, description = line.strip().split("|")  # Split directory path and description
    destination_file = os.path.join(destination_dir, f"{count}.wav")  # Create destination file path
    shutil.copyfile(directory, destination_file)  # Copy file to destination directory
    count += 1

print("Files copied successfully!")