def align_path(path_to_data, path_to_filelist):
    lines = []
    with open(path_to_filelist, "r", encoding="utf-8") as file:
        for line in file:
            dirs = line.split("/")
            line = dirs[len(dirs) - 1]
            line = path_to_data + line
            lines.append(line)
    with open(path_to_filelist, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line)
            
multilingual_data_path = "../data/"
multilingual_train_filelist_path = "./multilingual_train_filelist.txt"
multilingual_test_filelist_path = "./multilingual_test_filelist.txt"
multilingual_val_filelist_path = "./multilingual_val_filelist.txt"
align_path(multilingual_data_path, multilingual_train_filelist_path)
align_path(multilingual_data_path, multilingual_test_filelist_path)
align_path(multilingual_data_path, multilingual_val_filelist_path)
