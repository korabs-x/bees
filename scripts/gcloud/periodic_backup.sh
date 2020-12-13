#!/bin/bash

cd "$HOME"

set -u
PREFIX=$1
backupdir="${HOME}/backup-${PREFIX}"

while true; do
    if [[ -d $backupdir ]]; then
        gsutil cp -n -r "$backupdir" "gs://bee_living_sensor_obj_detection"
        gsutil cp "$backupdir/yolov4-custom_last.weights" "gs://bee_living_sensor_obj_detection/backup-${PREFIX}"
    fi
    sleep 15m
done
