"""
Give this script a path to a directory and it will create you an empty text file
for every image file in that directory.

This script won't override any files that already exist
"""
import sys
import os
import pathlib

args = sys.argv[1:]
assert len(args)==1
dir_path = args[0]
assert pathlib.Path(dir_path).is_dir()

def is_image(dir_entry):
    if dir_entry.is_file(follow_symlinks=False):
        fname = dir_entry.name
        if fname.endswith('.jpg') or fname.endswith('.jpeg'):
            return True
    return False

# not a good function, doesn't handle edge cases
def get_txtname_for_imgname(img_name):
    name = img_name.replace('.jpg', '').replace('.jpeg', '')
    return name + '.txt'

for dir_entry in os.scandir(dir_path):
    if is_image(dir_entry):
        touch_path = os.path.join(dir_path, get_txtname_for_imgname(dir_entry.name))
        # make sure to not overwrite anything
        if not pathlib.Path(touch_path).exists():
            pathlib.Path(touch_path).touch()

