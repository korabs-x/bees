import tensorflow as tf
from tensorflow import ConfigProto, Session, import_graph_def, GraphDef, gfile
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statistics
from tqdm import tqdm

#%matplotlib inline

MODEL_PATH = 'model_archive/models/all_except_erlen.pb'
VIDEO_PATH = 'videos/mucho_bees.mp4'
IMAGE_PATH = 'test_images'

def get_frozen_graph(graph_file):
    with gfile.GFile(graph_file, "rb") as f:
    #with gfile.FastGFile(graph_file, "rb") as f:
        #graph_def = tf.GraphDef()
        graph_def = GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

trt_graph = get_frozen_graph(MODEL_PATH)
input_names = ['image_tensor']
tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = Session(config=tf_config)
import_graph_def(trt_graph, name='')
tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')


def CNN_get_boxes_for_frame(image):
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
        if scores[i] < 0.8:
            continue
        box = boxes[i] * ([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
        box = np.round(box).astype(int)

        bboxes.append([box[1], box[0], box[3] - box[1], box[2] - box[0]])
    return bboxes


def plot_image_and_boxes_on_axis(image, boxes, ax, centers=False, trace=True, title=""):
    ax.imshow(image[:, :, ::-1])
    for i, box_dict in enumerate(boxes):
        # box = box_dict['location']
        box = box_dict
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], fill=False, lw=2, color='red')

        ax.add_patch(rect)
        ax.set_axis_off()
    ax.title.set_text(title)
    return ax


image_list = os.listdir(IMAGE_PATH)


ten_images = random.sample(image_list, 347)


image_arrays = []
for file in ten_images:
    img = cv2.imread(os.path.join(IMAGE_PATH, file))
    image_arrays.append(img)

nr_boxes = []

for img in tqdm(image_arrays):
    bboxes = CNN_get_boxes_for_frame(img)
    nr_boxes.append(len(bboxes))
    #fig, ax = plt.subplots(1, figsize=(20,10))
    #ax = plot_image_and_boxes_on_axis(img, bboxes, ax)
    #plt.show()

print("Max: {}".format(max(nr_boxes)))
print("Mean: {}".format(statistics.mean(nr_boxes)))