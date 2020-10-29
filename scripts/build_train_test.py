from pathlib import Path

train = Path("train.txt")
test = Path("test.txt")
l_dir = Path(".")


def is_bee_label_file(file):
    return file.suffix == ".txt" and "bee" in file.name


empty = 0
not_empty = 0
with open(train, "w") as train_f, open(test, "w") as test_f:
    for f in l_dir.iterdir():
        if is_bee_label_file(f):
            text = f.read_text()
            if text:
                not_empty = (not_empty + 1) % 5
                if not_empty == 0:
                    test_f.write(str(f.name).replace(".txt", ".jpg") + "\n")
                else:
                    train_f.write(str(f.name).replace(".txt", ".jpg") + "\n")
            else:
                empty = (empty + 1) % 5
                if empty == 0:
                    test_f.write(str(f.name).replace(".txt", ".jpg") + "\n")
                else:
                    train_f.write(str(f.name).replace(".txt", ".jpg") + "\n")
