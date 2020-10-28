from pathlib import Path
import sys

args = sys.argv[1:]
assert len(args) == 1
dir_path = Path(args[0])
assert dir_path.is_dir()

labeled_img_dir = dir_path / Path("labeled_imgs")

for file in dir_path.iterdir():
    if file.name.endswith(".txt"):
        text = file.read_text()
        if text:
            file.rename(labeled_img_dir / Path(file.name))
            img = Path(dir_path / file.name.replace(".txt", ".jpg"))
            if img.exists():
                img.rename(labeled_img_dir / Path(img.name))
