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


def get_frames(video_filename, n=float("inf"), nth=1, start=0):
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
            if (i % nth) == 0:
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
    print("Boxes", len(boxes))
    print(sum([len(b) for b in boxes]))

    for file_name in glob.glob("out_video/*.jpg"):
        os.remove(file_name)
    # frames = get_frames(video_filename, n=n, nth=nth)
    # print("Frames", len(frames))

    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
              '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
              '#000075', '#808080', '#ffffff', '#000000']

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
        tracker = Sort(min_hits=1, iou_threshold=0, max_age=1)
    # start tracking
    for frame_boxes in boxes:
        track_bboxes = np.array([[box['xmin'], box['ymin'], box['xmax'], box['ymax']] for box in frame_boxes]).astype(
            "int")
        track_objects = tracker.update(track_bboxes)
        if tracking_func == 'sort':
            # track_bboxes = [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in track_objects]
            track_objects = {d[4]: ((d[2] + d[0]) * 0.5, (d[3] + d[1]) * 0.5) for d in track_objects}
        tracking_objects.append(deepcopy(track_objects))
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


if __name__ == "__main__":
    # boxes = read_box_file('data/boxes.csv')
    boxes = get_tracking_boxes('data/annotations.xml')
    frames = get_frames('data/Doettingen_Hive_1_M_Rec_20200427_171545_540_M.mp4', start=48 * 25, n=12 * 25)
    tracking_objects = get_tracking_objects(boxes, 'sort')
    eval_tracking(boxes, tracking_objects)
    tracking_objects = get_tracking_objects(boxes, 'notsort')
    eval_tracking(boxes, tracking_objects)
    # plot_tracking(frames, boxes, tracking_objects)
    """
    frames = get_frames('data/Doettingen_Hive_1_M_Rec_20200427_171545_540_M.mp4', n=float("inf"), nth=1)
    for i, frame in enumerate(frames):
        print(i)
        if i < 48*25:
            continue
        if i >= 25*60:
            break
        cv2.imwrite("out_video/file%02d.jpg" % (i-48*25), frame)
    """
