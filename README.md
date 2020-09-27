# bees
Data Science Lab 2020 - Bees

## Potential Tasks
- Improve tracking / bee segmentation
- Improve pollen detection
- Pollen color clustering
- Movement pattern detection / clustering

## Ideas
- Bee carries pollen ⇒  Always incoming 
- What's the best way to get the colour of the pollen? There has to be some good non-ML method (eg [https://ieeexplore.ieee.org/abstract/document/8354145](this one))

## Extra datasets
- [https://github.com/piperod/PollenDataset] (Pollen Dataset): More cropped images of bees labeled per instance P/NP
- [https://www.kaggle.com/kport354041/honeybee-positions] (Honeybee positions): Images the whole hive for tracking bees, includes position & orientation of bees. Might be able to use it for direction prediction, have to crop images of individual bees first.
- [https://www.kaggle.com/jenny18/honey-bee-annotated-images?] (Honeybee annotated images): 5100 bee images with annotations including pollen (only 18 with pollen). Any ideas if this is useful? I (Korbi) cannot think of how.
- [https://www.kaggle.com/jerzydziewierz/bee-vs-wasp] (Bee vs wasp): 3183 images of bees on flowers. Probably not usable for us.
- [https://groups.oist.jp/bptu/honeybee-tracking-dataset] (Honeybee tracking): Segmented videos of bees inside the hive. Probably not usable for us.

## NAS
[https://drive.google.com/file/d/1-Y--t0fuEOe46c2S9qMFx2HPsCTu8ubN/view?usp=sharing](PDF Classification Guideline)


1. *Backups*: Who knows
2. ~~bee_jsons: empty~~
3. *classified_bees*: Pictures of indiv bees, cropped to 200x200. Labelled through filename into P (pollen), NP (no pollen), M (missclassified ). Three different days, hive united queens. Total: ~1200 images.
4. *extracted_bees*: under images_old ~6500 unlabeled, cropped images of indiv bees
5. *images*: under tracker_output ~14000 unlabled, cropped images of indiv bees
6. *models*: Two model files detector and classifier.
7. *object_detection*: Gabriela + Pascal stuff. Everything else (except videos) is from last year
    1. *images*: Old stuff that Gabriela will delete
    2. *labeling_new*: Different hives, bee bounding boxes labeled via .txt and .xml, ~4300. This is what Daniela + Pascal worked with
    3. Some further folders with their YOLO stuff
8. *pollen_frames*:
    1. *images*: Full images of hive entry, some of them classified into P/NP but no bounding boxes. Prob unusable.
        1. ~~bees: empty~~
        2. *classified*: 1363 full (non cropped!) images classified into Pollen (incl colour)/ no Pollen.
        3. ~~from_nas: empty~~
        4. *night*: 2500 unlabeled, nightly pictures of hive entry
        5. *no_bees*: 800 pictures of empty hive entry. Look v similar
        6. *outliers*: 9 pictures of blurry hive entry
        7. *tmp_bee* / *tmp_classified* / *tmp_nobee*: Content looks exactly like bee / classified / no_bees folders. What is this?
    2. *images_old*: Full images of United Queens labeled p/np
9. *videos*: total 5661 videos. Eg Doettingen Hive 204 normal vids, 204 slow mo.


