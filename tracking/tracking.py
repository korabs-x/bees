# given a video, boxes and a tracking function, plot tracking
import cv2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import subprocess
import glob
from pyimagesearch.centroidtracker import CentroidTracker
import numpy as np
from sort import Sort
import xml.etree.ElementTree as ET
import motmetrics as mm
from time import time
from copy import deepcopy
from podm.podm import get_pascal_voc_metrics, BoundingBox, MetricPerClass
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import fclusterdata
from sklearn.metrics import homogeneity_score


def get_tracking_boxes(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    frame_boxes = []
    for track in root.findall('track'):
        for box in track.findall('box'):
            if box.attrib['outside'] == '1':
                continue
            frame = int(box.attrib['frame'])
            if len(frame_boxes) <= frame:
                frame_boxes += [[] for _ in range(frame - len(frame_boxes) + 1)]
            bbox = {'xmin': float(box.attrib['xtl']),
                    'ymin': float(box.attrib['ytl']),
                    'xmax': float(box.attrib['xbr']),
                    'ymax': float(box.attrib['ybr']),
                    'id': track.attrib['id']}
            frame_boxes[frame].append(bbox)
    return frame_boxes


def get_frames(video_filename, n=float("inf"), nth=1, start=0, skip=[]):
    print("Get frames")
    # Opens the Video file
    # frames = []
    cap = cv2.VideoCapture(video_filename)
    i = 0
    while cap.isOpened() and (i - start) // nth < n:
        ret, frame = cap.read()
        if i >= start:
            if not ret:
                break
            if ((i - start) % nth) == 0 and ((i - start) % nth) not in skip:
                print(i)
                yield frame
                # frames.append(frame)
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return


def read_box_file(box_filename):
    df = pd.read_csv(box_filename)
    boxes = []
    frame_boxes = None
    image_idx = None
    for idx, box in df.iterrows():
        if box['image'] != image_idx:  # or True:
            if frame_boxes is not None:
                boxes.append(frame_boxes)
                # print(box['image'], len(boxes))
            frame_boxes = []
            image_idx = box['image']
        frame_boxes.append(box)
    return boxes


def read_boxes_from_yolo_output(path, height=1920, width=2560):
    boxes = []
    for filename in sorted([x for x in os.listdir(path) if x.endswith('.txt')], key=lambda name: int(name[:-4])):
        frame_boxes = []
        with open(os.path.join(path, filename)) as f:
            lines = f.read().splitlines()
            for line in lines:
                info = line.split(' ')
                boxwidth = float(info[3]) * width
                boxheight = float(info[4]) * height
                xmin = int(float(info[1]) * width - boxwidth * 0.5)
                ymin = int(float(info[2]) * height - boxheight * 0.5)
                xmax = int(boxwidth) + xmin
                ymax = int(boxheight) + ymin
                frame_boxes.append({
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })
        boxes.append(frame_boxes)
    return boxes


def plot_image_on_axis(image, ax):
    ax.imshow(image[:, :, ::-1])
    return ax


def plot_boxes_on_axis(boxes, ax, title="", color='blue'):
    for i, box_dict in enumerate(boxes):
        # box = box_dict['location']
        box = box_dict
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], fill=False, lw=2, color=color)

        ax.add_patch(rect)
        ax.set_axis_off()
    ax.title.set_text(title)
    return ax


def plot_tracking(frames, boxes, tracking_objects=None):
    out_path = 'out_video'
    print("Boxes", len(boxes))
    print(sum([len(b) for b in boxes]))

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for file_name in glob.glob(f'{out_path}/*.jpg'):
        os.remove(file_name)

    for i, (frame_boxes, frame, track_objects) in enumerate(zip(boxes, frames, tracking_objects)):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        plot_bboxes = []
        for box in frame_boxes:
            plot_bboxes.append([box['xmin'], box['ymin'], box['xmax'] - box['xmin'], box['ymax'] - box['ymin']])
        ax = plot_image_on_axis(frame, ax)
        ax = plot_boxes_on_axis(plot_bboxes, ax, color='blue')
        if tracking_objects is None:
            pass
        else:
            # plot tracking ids
            for (objectID, centroid) in track_objects.items():
                plt.text(centroid[0] - 20, centroid[1], str(int(objectID)), color='red', fontsize=12, fontweight='bold')
        print("savefig", i)
        plt.savefig("out_video/file%02d.jpg" % i, bbox_inches='tight', pad_inches=0)
        plt.close()

    os.chdir("out_video")
    subprocess.call([
        'ffmpeg', '-framerate', '25', '-i', 'file%02d.jpg', '-r', '25', '-pix_fmt', 'yuv420p',
        'video.mp4'
    ])
    os.chdir("..")


def get_tracking_objects(boxes, tracking_func):
    tracking_objects = []
    # set tracker
    tracker = None
    if tracking_func == 'notsort':
        tracker = CentroidTracker()
    elif tracking_func == 'sort':
        tracker = Sort(min_hits=1, iou_threshold=0, max_age=2)
    # start tracking
    i = 0
    for frame_boxes in boxes:
        track_bboxes = np.array([[box['xmin'], box['ymin'], box['xmax'], box['ymax']] for box in frame_boxes]).astype(
            "int")
        track_objects = tracker.update(track_bboxes)
        if tracking_func == 'sort':
            # track_bboxes = [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in track_objects]
            track_objects = {d[4]: ((d[2] + d[0]) * 0.5, (d[3] + d[1]) * 0.5) for d in track_objects}
        tracking_objects.append(deepcopy(track_objects))
        i += 1
    return tracking_objects


def get_distance(boxes, centroids):
    box_centroids = [[(box['xmin'] + box['xmax']) * 0.5, (box['ymin'] + box['ymax']) * 0.5] for box in boxes]
    distance = [[np.sqrt((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2) for c1 in centroids] for c0 in box_centroids]
    return distance


def eval_tracking(boxes, tracking_objects):
    acc = mm.MOTAccumulator(auto_id=True)
    for frame_boxes, frame_objects in zip(boxes, tracking_objects):
        ground_truth = np.array([box['id'] for box in frame_boxes]).astype('int')
        obj_ids = [key for key, _ in frame_objects.items()]
        prediction = np.array(obj_ids).astype('int')
        dist = get_distance(frame_boxes, [frame_objects[key] for key in obj_ids])
        acc.update(
            ground_truth,
            prediction,
            dist
        )
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'precision', 'recall', 'idp', 'idr', 'idf1'], name='acc')
    print(summary)


def eval_obj_det(pred_boxes, true_boxes):
    print("Eval obj det")
    gt_BoundingBoxes = [BoundingBox('{}'.format(i), 'bee', box['xmin'], box['ymin'], box['xmax'], box['ymax'], score=1)
                        for i, boxes in
                        enumerate(true_boxes) for box in boxes]
    pd_BoundingBoxes = [
        BoundingBox('{}'.format(i), 'bee', box['xmin'], box['ymin'], box['xmax'], box['ymax'], score=0.9)
        for i, boxes in
        enumerate(pred_boxes) for box in boxes]
    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
    print("Precision", results['bee'].precision[-1])
    print("Recall", results['bee'].recall[-1])
    print("Mean Average Precision", MetricPerClass.get_mAP(results))


if __name__ == "__main__":
    use_new_model = True
    use_sort = True
    create_tracking_video = True

    # read predicted and true bounding boxes
    if use_new_model:
        boxes = read_boxes_from_yolo_output('data/boxes171545_newmodel')
    else:
        boxes = read_box_file('data/boxes171545.csv')
    true_boxes = get_tracking_boxes('data/annotations171545.xml')

    # apply tracking algorithms
    tracking_objects_sort = get_tracking_objects(boxes, 'sort')
    print("\nEvaluate sort tracking on predicted boxes")
    eval_tracking(true_boxes, tracking_objects_sort)

    tracking_objects_old = get_tracking_objects(boxes, 'notsort')
    print("\nEvaluate original tracking on predicted boxes")
    eval_tracking(true_boxes, tracking_objects_old)

    # create tracking video
    if create_tracking_video:
        frames = get_frames('data/Doettingen_Hive_1_M_Rec_20200427_171545_540_M.mp4', start=48 * 25, n=12 * 25)
        plot_tracking(frames, boxes, tracking_objects_sort)
