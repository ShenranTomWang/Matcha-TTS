def remove_numbers(in_path):
    lines = []
    with open(in_path, "r") as file:
        for line in file:
            sentence: str = line.split("|")[1]
            if not any(char.isdigit() for char in sentence):
                lines.append(line)
    with open(in_path, "w") as file:
        for line in lines:
            file.write(line)

train_path = "./ojibwe_NJ_train_filelist.txt"
test_path = "./ojibwe_NJ_test_filelist.txt"
val_path = "./ojibwe_NJ_val_filelist.txt"

remove_numbers(train_path)
remove_numbers(test_path)
remove_numbers(val_path)