import sys
from pathlib import Path

args = sys.argv[1:]
assert len(args) == 1, "Need the name of the directory!"
dir_path = args[0]
l_dir = Path(dir_path)
assert l_dir.is_dir()

train = l_dir / Path("train.txt")
test = l_dir / Path("test.txt")


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
