"""
Give this script a path to a directory and it will look at all the label
files (the .txt's associated with an img)
It will check whether there is something (eg Pollen) annotated in the label file
and output statistics
"""
import sys
from pathlib import Path

args = sys.argv[1:]
assert len(args) == 1

dir_path = args[0]
directory =Path(dir_path)
assert directory.is_dir()


def is_image(dir_entry):
    if dir_entry.is_file():
        fname = dir_entry.name
        if fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".png"):
            return True
    return False


# not a good function, doesn't handle edge cases
def get_txtname_for_imgname(img_name):
    name = img_name.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
    return name + ".txt"


stats = {"not yet annotated": 0, "no Label": 0}
for dir_entry in directory.iterdir():
    if is_image(dir_entry):
        annot_file = Path(directory / get_txtname_for_imgname(dir_entry.name))
        if not annot_file.exists():
            stats["not yet annotated"] += 1
        else:
            text = annot_file.read_text()
            if text:
                stats[f"Label {text[0]}"] = stats.get(f"Label {text[0]}", 0) + 1
            else:
                stats["no Label"] += 1

print("Num imgs in folder:", sum(v for v in stats.values()))
print()
print("\n".join(f"{key}: {value}" for key, value in stats.items()))
