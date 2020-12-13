#!/bin/bash

set -euo pipefail

if (( $# != 1)); then
    echo "run using $0 <PREFIX>"
    exit 1
fi

PREFIX="$1"
GSBUCKET="gs://bee_living_sensor_obj_detection"

# sign into gcloud
gcloud auth activate-service-account --key-file=bee-living-sensor-da7ae770e263.json

#
# DATA PREPARATION
#
# download all datasets from gstorage
mkdir -p "$HOME/data/" "$HOME/backup-${PREFIX}/"
gsutil ls -r "${GSBUCKET}/"** | grep .zip > ~/available_datasets.txt
gsutil cp -n "$(cat available_datasets.txt)" ~/data/
cd ~/data
unzip -q \*.zip
cd

# take all image files in validate/ or test/ folders and write full path to test.txt
find data/ -type f -name \*.jpg | grep -E "(/validate/|/test/)" | xargs realpath > test.txt
# take all other image files and write full path to test.txt
find data/ -type f -name \*.jpg | grep -E -v "(/validate/|/test/)" | xargs realpath > train.txt

#
# SETUP OPENCV
#
sudo apt install -y cmake g++ wget unzip
wget -N -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
unzip -q opencv.zip
mkdir -p build && cd build
cmake  -D OPENCV_GENERATE_PKGCONFIG=YES ../opencv-master
cmake --build .
make && sudo make install
cd

#
# SETUP DARKNET
#
if [[ ! -d darknet ]]; then
    git clone https://github.com/AlexeyAB/darknet darknet/
fi
cd darknet
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
make
chmod +x darknet

# download pretrained weights
wget -N https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
gsutil cp "${GSBUCKET}/yolov4-custom.cfg" .
sed -i 's:max_batches = 6000:max_batches = 20000:' yolov4-custom.cfg

# make obj.names and obj.data
printf "bee\n" > obj.names
printf 'classes = 2\ntrain = %s/train.txt\nvalid = %s/test.txt\nnames = obj.names\n' "$HOME" "$HOME" > obj.data
printf 'backup = %s/backup-%s/\n' "$HOME" "$PREFIX" >> obj.data

# run training
./darknet detector train obj.data yolov4-custom.cfg yolov4.conv.137 -dont_show -map
cd

#
# UPLOAD WEIGHTS
#
gsutil cp -n -r "backup-${PREFIX}" "$GSBUCKET"

#
# CALC SCORES
#
cd darknet
scorefile=scores-$PREFIX.txt
if [[ ! -f "$scorefile" ]]; then
    for weights in ~/backup-${PREFIX}; do
        printf 'Current: %s\n\n' "$weights" >> "$scorefile"
        ./darknet detector map obj.data yolov4-custom.cfg "$weights" >> "$scorefile"
    done
fi
gsutil cp "$scorefile" "$GSBUCKET"
