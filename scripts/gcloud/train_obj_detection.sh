#!/bin/bash

set -euo pipefail

if (( $# != 1)); then
    echo "run using $0 <PREFIX>"
    exit 1
fi

PREFIX="$1"
GSBUCKET="gs://bee_living_sensor_obj_detection"

#
# SETUP GSUTILS
#
if ! command -v gcloud &> /dev/null ; then
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-320.0.0-linux-x86_64.tar.gz
    tar -xf google-cloud-sdk-320.0.0-linux-x86_64.tar.gz
    GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
    GSUTIL="$HOME/google-cloud-sdk/bin/gsutil"
else
    GCLOUD="gcloud"
    GSUTIL="gsutil"
fi
$GCLOUD auth activate-service-account --key-file=bee-living-sensor-da7ae770e263.json

#
# DATA PREPARATION
#
# download all datasets from gstorage
mkdir -p "$HOME/data/" "$HOME/backup-${PREFIX}/"
$GSUTIL ls -r "${GSBUCKET}/"** | grep .zip > ~/available_datasets.txt
$GSUTIL cp -n $(cat available_datasets.txt) ~/data/
cd ~/data
unzip -qu \*.zip
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
unzip -qu opencv.zip
mkdir -p build && cd build
cmake -D OPENCV_GENERATE_PKGCONFIG=YES ../opencv-master
cmake --build .
make && sudo make install
# Without these darknet doesn't find openCV, but I don't know why they work
sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
cd

#
# SETUP DARKNET
#
if [[ ! -d darknet ]]; then
    git clone https://github.com/AlexeyAB/darknet darknet/
    cd darknet
    sed -i 's/OPENCV=0/OPENCV=1/' Makefile
    sed -i 's/GPU=0/GPU=1/' Makefile
    sed -i 's/CUDNN=0/CUDNN=1/' Makefile
    sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
    make
    chmod +x darknet
    $GSUTIL cp "${GSBUCKET}/yolov4-custom.cfg" .
    # sed -i 's:max_batches = 6000:max_batches = 6000:' yolov4-custom.cfg
    cd
fi
cd darknet

#
# DOWNLOAD WEIGHTS
#
gsyolo_url=$GSBUCKET/backup-${PREFIX}/yolov4-custom_last.weights
if $GSUTIL -q stat "$gsyolo_url"; then
    $GSUTIL cp -n "$gsyolo_url" .
    weightsfile=yolov4-custom_last.weights
else
    wget -N https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
    weightsfile=yolov4.conv.137
fi

# make obj.names and obj.data
printf "bee\n" > obj.names
printf 'classes = 2\ntrain = %s/train.txt\nvalid = %s/test.txt\nnames = obj.names\n' "$HOME" "$HOME" > obj.data
printf 'backup = %s/backup-%s\n' "$HOME" "$PREFIX" >> obj.data

# start background backup process
screen -d -m -S backup_process bash "$HOME/periodic_backup.sh" "$PREFIX"

# run training
./darknet detector train obj.data yolov4-custom.cfg $weightsfile -dont_show
cd

#
# UPLOAD WEIGHTS
#
$GSUTIL cp -n -r "backup-${PREFIX}" "$GSBUCKET"

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
$GSUTIL cp "$scorefile" "$GSBUCKET"

sudo shutdown -h
