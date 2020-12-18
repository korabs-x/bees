#!/bin/bash

set -xeuo pipefail

if (( $# != 2)); then
    echo "run using $0 <PREFIX> <WEIGHTSFILE>"
    exit 1
fi

PREFIX=$1
WEIGHTSFILE=$2

cd "$HOME/darknet" || exit
printf "bee\n" > obj.names
printf 'classes = 2\ntrain = %s/train.txt\nvalid = %s/test.txt\nnames = obj.names\n' "$HOME/darknet" "$HOME/darknet" > obj.data
printf 'backup = %s/backup-%s\n' "$HOME" "$PREFIX" >> obj.data
scorefile="scores-each-${PREFIX}.txt"
for hive in $(find "$HOME/data" -mindepth 1 -maxdepth 1 -type d); do
    find "$hive" -type f | grep -E "(jpg|jpeg|png)" | grep -E "(/validate/|/test/)" | xargs realpath > "test.txt"
    printf '\n\nHIVE: %s\n' "$hive" | tee --append "$scorefile"
    ./darknet detector map obj.data yolov4-custom.cfg "$HOME/$WEIGHTSFILE" -thresh 0.25 >> "$scorefile"
done
cd "$HOME"
