#!/bin/bash

cd "$HOME"

set -u
PREFIX=$1
backupdir="${HOME}/backup-${PREFIX}"

while true; do
    if [[ -d $backupdir ]]; then
        gsutil cp -n -r "$backupdir" "gs://bee_living_sensor_obj_detection"
    fi
    sleep 5m
done
