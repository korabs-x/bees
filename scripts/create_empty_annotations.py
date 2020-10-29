"""
Give this script a path to a directory and it will create you an empty text file
for every image file in that directory.

This script won't override any files that already exist

Usecase: You have labelled all the P's and need empty .txt's for the NPs
"""
import sys
from pathlib import Path

args = sys.argv[1:]
assert len(args) == 1
dir_path = Path(args[0])
assert dir_path.is_dir()


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


for dir_entry in dir_path.iterdir():
    if is_image(dir_entry):
        touch_path = dir_path / Path(get_txtname_for_imgname(dir_entry.name))
        # make sure to not overwrite anything
        if not touch_path.exists():
            # create new completely empty file
            touch_path.touch()
