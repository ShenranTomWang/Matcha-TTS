def align_path(path_to_data, path_to_filelist):
    lines = []
    with open(path_to_filelist, "r") as file:
        for line in file:
            line = path_to_data + line
            lines.append(line)
    with open(path_to_filelist, "w") as file:
        for line in lines:
            file.write(line)
            
multilingual_data_path = "/project/6080355/shenranw/data/"
multilingual_train_filelist_path = "./mikmaw_train_filelist.txt"
multilingual_test_filelist_path = "./mikmaw_test_filelist.txt"
multilingual_val_filelist_path = "./mikmaw_val_filelist.txt"
align_path(multilingual_data_path, multilingual_train_filelist_path)
align_path(multilingual_data_path, multilingual_test_filelist_path)
align_path(multilingual_data_path, multilingual_val_filelist_path)
