# bees
Data Science Lab 2020 - Bees

## Docker
The Dockerfile builds openCV and YOLO.
To run the resulting image [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker) needs to be installed.

Before we had the Docker images we ran the models directly on the machines. 
The scripts we used can be found in the `scripts/gcloud` and `scripts/azure` directories.
We used the yolo.cfg files as is, only ever adjusting the input image dimensions directly in the scripts.

## Overview over data
### Bee object detection
We had data on the hives: Chueried Hempbox, Chueried Hive01, Clemens Red, Doettingen Hive1, Echolinde, Erlen Hive 04 (diagonalview), Erlen Hive 04 (frontview), Erlen Hive 04 (smartphone), Erlen Hive 11, Froh 14, Froh 23, Unitied (Unified) Queens.

A previous version of the data didn't have blurry (flying) bees labeled, hence our data is called _blurred-labeled.zip_ often.
All of this data we went through manually and checked every single image, hence the data quality should be good

We then went on MTurk to get further data.
This one we didn't check manually, hence the data quality ended up being much worse. 
We only slightly cleaned the data (removed some unlabeled images).
It's kept separate from the 'good' data in the `bee_data_mtruk_cleaned.zip` file.

### Pollen Detection
All of our Pollen Data is in `pollen_data_complete.zip`.
It contains a bunch of datasets, one we got online (PollenDataset), one we got from the previous team (DSL2019).
The 'hybrid' datasets refer to running an early Pollen detection model with low threshold over unlabeled data to get candidates for images containing Pollen.
Hence those datasets will have a higher P/NP percentage.

## Overview over trained models
### Object detection
Models were trained in three input dimensions: small (224), medium (416), large (608).
The resulting model size is the same (~250MB), the only thing that differs is how to images are resizes before being run through the model.
The 'small' models operate on images of size 224x224 and are roughly twice as fast as the large model.

- backup-train-all: small model fit on all available data (no MTurk)
- backup-train-all-medium: medium model fit on all available data (no MTurk)
- backup-train-gb-settings: large model fit on all available data (no MTurk)
- only & except models: Small models either fit on just one hive, or on all hives except that one (no MTurk)
- MTurk large pretrain: large model fit just on the MTurk data
- Mturk-finetune: MTurk large pretrain model, finetuned for 6000 iterations on all the data that wasn't MTurk.
### Pollen Detection
We trained two models: Large (250MB) and tiny (22MB).
Both run on images of size 224x224.
The large model has it's config file in `yolov4-custom.cfg`, the tiny one is `yolov4-tiny-custom.cfg`.
The tiny one is worse, but reaches ~10x the framerate of the large one.
