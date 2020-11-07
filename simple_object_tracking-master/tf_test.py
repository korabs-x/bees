import sys
import tensorflow as tf
# from tensorflow import ConfigProto, Session, import_graph_def, GraphDef, gfile
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import sqlite3
from shapely.geometry import Polygon
import numpy as np
from copy import deepcopy
import pandas as pd
import seaborn as sns


MODEL_PATH = '/Users/pascal/Coding/MP_bees/training_01_05/frozen_inference_graph.pb'
DB_PATH = '/Users/pascal/Coding/MP_bees/simple_object_tracking/bees.db'
IMAGE_PATH = "/Users/pascal/Coding/MP_bees/images/test"
PLOT = False
image_list = []
for filename in os.listdir(IMAGE_PATH):
    if not filename.endswith('.jpg'): continue
    fullname = os.path.join(IMAGE_PATH, filename)
    image_list.append(fullname)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
print('running with TensorFlow version {}'.format(tf.__version__))
def get_frozen_graph(graph_file):
    with tf.gfile.GFile(graph_file, "rb") as f:
    #with gfile.FastGFile(graph_file, "rb") as f:
        #graph_def = tf.GraphDef()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

trt_graph = get_frozen_graph(MODEL_PATH)
input_names = ['image_tensor']
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')
tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')


def CNN_get_boxes_for_frame(file,image):

    predicted = {}
    predicted[file] = {}
    predicted[file]['boxes'] = []
    predicted[file]['scores'] = []
    bboxes = []

    # image = frame

    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections],
                                                         feed_dict={
                                                             tf_input: image[None, ...]
                                                         })
    boxes = boxes[0]
    scores = scores[0]
    classes = classes[0]
    num_detections = int(num_detections[0])

    for i in range(scores.shape[0]):
        if scores[i] < 0.7:
            continue
        box = boxes[i] * ([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
        box = np.round(box).astype(int)
        y_min = box[0]
        x_min = box[1]
        y_max = box[2]
        x_max = box[3]


        predicted[file]['boxes'].append([x_min, y_min, x_max, y_max])
        predicted[file]['scores'].append(scores[i])
    return predicted



def plot_image_and_boxes_on_axis(image, boxes, ax, centers=True, trace=True, title="",color='blue'):
    ax.imshow(image[:, :, ::-1])
    for i, box_dict in enumerate(boxes):
        # box = box_dict['location']
        box = box_dict
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], fill=False, lw=2, color=color)

        ax.add_patch(rect)
        ax.set_axis_off()
    ax.title.set_text(title)
    return ax

def calculate_iou(box_1, box_2):
    # if box_1[4] not in range(int(0.9*box_2[4]),int(1.1*box_2[4])) and box_1[5] not in range(int(0.9*box_2[5]),int(1.1*box_2[5])):
    #     return None
    box_1 = [[box_1[0],box_1[3]],[box_1[1],box_1[3]],[box_1[1],box_1[2]],[box_1[0],box_1[2]]]
    box_2 = [[box_2[0],box_2[3]],[box_2[1],box_2[3]],[box_2[1],box_2[2]],[box_2[0],box_2[2]]]
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def get_model_scores(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.
    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
    """
    model_score={}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_score.keys():
                model_score[score]=[img_id]
            else:
                model_score[score].append(img_id)
    return model_score


def calc_iou(gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_bbox

    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct", x_topleft_p, x_bottomright_p, y_topleft_p,
                             y_bottomright_p)

    # if the GT bbox and predcited BBox do not overlap then iou=0
    if (x_bottomright_gt < x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox

        return 0.0
    if (
            y_bottomright_gt < y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox

        return 0.0
    if (
            x_topleft_gt > x_bottomright_p):  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox

        return 0.0
    if (
            y_topleft_gt > y_bottomright_p):  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox

        return 0.0

    GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
    Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (y_bottomright_p - y_topleft_p + 1)

    x_top_left = np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])

    intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)

    return intersection_area / union_area


def calc_precision_recall(image_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_positive=0
    false_positive=0
    false_negative=0
    precision = 0
    recall = 0
    for img_id, res in image_results.items():
        true_positive +=res['true_positive']
        false_positive += res['false_positive']
        false_negative += res['false_negative']
        try:
            precision = true_positive/(true_positive+ false_positive)
        except ZeroDivisionError:
            precision=0.0
        try:
            recall = true_positive/(true_positive + false_negative)
        except ZeroDivisionError:
            recall=0.0
    return (precision, recall)


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []

    for ipb, pred_box in enumerate(pred_boxes):
        # print(ipb, pred_box)

        for igb, gt_box in enumerate(gt_boxes):
            # print(igb, gt_box)
            iou = calc_iou(gt_box, pred_box)

            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': len(pred_boxes), 'false_negative': fn}
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in iou_sort:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
        output = {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    return output



def get_avg_precision_at_iou(image,gt_boxes, pred_bb, iou_thr=0.5):
    model_scores = get_model_scores(pred_bb)
    sorted_model_scores = sorted(model_scores.keys())
    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_bb.keys():
        arg_sort = np.argsort(pred_bb[img_id]['scores'])
        pred_bb[img_id]['scores'] = np.array(pred_bb[img_id]['scores'])[arg_sort].tolist()
        pred_bb[img_id]['boxes'] = np.array(pred_bb[img_id]['boxes'])[arg_sort].tolist()


    pred_boxes_pruned = deepcopy(bboxes)


    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    img_ids = []
    model_score_thr = 0
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores):
        # On first iteration, define img_results for the first time:
        print("Model score : ", model_score_thr)
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores[model_score_thr]
    for img_id in img_ids:

        gt_boxes_img = gt_boxes[img_id]
        box_scores = pred_boxes_pruned[img_id]['scores']
        start_idx = 0

        for score in box_scores:
            if score <= iou_thr:
                pred_boxes_pruned[img_id]
                start_idx += 1
            else:
                break
                # Remove boxes, scores of lower than threshold scores:
        pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
        pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]
        # Recalculate image results for this image
        img_results[img_id] = get_single_image_results(gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr=0.5)
        # calculate precision and recall
    if len(img_results) == 0 and len(pred_bb[img_id]['boxes']) == 0 and len(gt_boxes[img_id]) != 0:
        return {img_id:{'true_positive': 0, 'false_positive': 0, 'false_negative': len(gt_boxes[img_id])}}
    prec, rec = calc_precision_recall(img_results)
    precisions.append(prec)
    recalls.append(rec)
    model_thrs.append(model_score_thr)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls > recall_level).flatten()
            prec = max(precisions[args])
            # print(recalls, "Recall")
            # print(recall_level, "Recall Level")
            # print(args, "Args")
            # print(prec, "precision")
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)
    return img_results

    # return {'hive':image,'avg_prec': avg_prec,'precisions': precisions[0],'recalls': recalls[0],'model_thrs': model_thrs[0]}




image_arrays = []
results = []
print("Extracting images...")

for i in tqdm(range(len(image_list))):
    img = cv2.imread(image_list[i])
    image_arrays.append(img)

print("Working on images...")
for i in tqdm(range(len(image_list)),position=0,leave=True):
    filename = image_list[i].split('/')[-1]
    bboxes = CNN_get_boxes_for_frame(filename, image_arrays[i])

    c.execute("select xmin,ymin,xmax,ymax from ground_truth where filename = '{}'".format(filename))
    ground_truth = c.fetchall()

    gt_bboxes = {}
    gt_bboxes[filename] = [list(elem) for elem in ground_truth]
    if PLOT:
        fig, ax = plt.subplots(1, figsize=(20,10))
        plot_bboxes = []
        for box in bboxes[filename]['boxes']:
            plot_bboxes.append([box[0], box[1], box[2]-box[0], box[3]-box[1]])
        plot_gt_bboxes = []
        for box in gt_bboxes[filename]:
            plot_gt_bboxes.append([box[0], box[1], box[2]-box[0], box[3]-box[1]])
        ax = plot_image_and_boxes_on_axis(image_arrays[i], plot_bboxes, ax,color='blue')
        ax = plot_image_and_boxes_on_axis(image_arrays[i], plot_gt_bboxes, ax,color='green')
        plt.savefig("predicted_actual_"+filename)
    results.append(get_avg_precision_at_iou(filename,gt_bboxes,bboxes))



output = pd.DataFrame()
for i in results:
    d = {}
    for k,v in i.items():
        d['frame'] = k
        for kk,vv in v.items():
            d[kk] = vv
    output = output.append(d,ignore_index=True)

output = output[['frame', 'true_positive','false_positive','false_negative']]

total_true_positive = output['true_positive'].sum()
total_false_negative = output['false_negative'].sum()
total_false_positive = output['false_positive'].sum()

all_names = []
for name in output.frame:
    all_names.append(name.split('_')[0])

for u_name in list(set(all_names)):
    print(u_name)
    filtered_df = output.loc[output['frame'].str.contains(u_name)]
    true_positive = filtered_df['true_positive'].sum()
    print("TP: {}".format(true_positive))
    false_negative = filtered_df['false_negative'].sum()
    print("FN: {}".format(false_negative))
    false_positive = filtered_df['false_positive'].sum()
    print("FP: {}".format(false_positive))
    print("{} images".format(len(filtered_df)))
    print("Precision: {}".format(true_positive/(true_positive+false_positive)))
    print("Recall: {}".format(true_positive / (true_positive + false_negative)))
    print("_____________________________________________________________")
print("Model name: {}".format(MODEL_PATH.split('/')[-2]))
print("Overall TP: {}".format(total_true_positive))
print("Overall FN: {}".format(total_false_negative))
print("Overall FP: {}".format(total_false_positive))
print("{} images".format(len(output)))
print("Overall Precision: {}".format(total_true_positive/(total_true_positive+total_false_positive)))
print("Overall Recall: {}".format(total_true_positive / (total_true_positive + total_false_negative)))
print("_____________________________________________________________")
print("_____________________________________________________________")



# for p in bboxes[filename]['boxes']:
#     for gt in gt_bboxes[filename]:
#         iou = calc_iou(gt,p)
#         if iou > 0.5:
#             print("GT: {} P: {} IOU: {}".format(gt,p,iou))