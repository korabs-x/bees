import numpy as np

import cv2

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/pascal/Coding/MP_bees/tensorflow')
import tensorflow as tf
sys.path.insert(2, '/Users/pascal/Coding/MP_bees/tensorflow/models')
sys.path.insert(3, '/Users/pascal/Coding/MP_bees/tensorflow/models/research')
sys.path.insert(4, '/Users/pascal/Coding/MP_bees/tensorflow/models/research/object_detection')
sys.path.insert(5, '/Users/pascal/Coding/MP_bees/simple_object_tracking')
from utils import label_map_util
from utils import visualization_utils as vis_util
from datetime import datetime
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import FPS
from imutils.video import VideoStream
import sqlite3
import dlib
import math

conn = sqlite3.connect('/Users/pascal/Coding/MP_bees/bees.db')
c = conn.cursor()

# c.execute("drop table coordinates")
# c.execute("drop table od_runs")

# c.execute("""create table coordinates (
#             run_id integer,
#             video text,
#             frame integer,
#             b_id integer,
#             x_start integer,
#             x_end integer,
#             y_start integer,
#             y_end integer,
#             x_center integer,
#             y_center integer,
#             confidence real
# )""")
# c.execute("""create table od_runs (
#             run_id integer,
#             model text,
#             threshold real,
#             date text
# )""")






# path to the frozen graph:
PATH_TO_FROZEN_GRAPH = '/Users/pascal/Coding/MP_bees/training_01_05/frozen_inference_graph.pb'
#PATH_TO_FROZEN_GRAPH = '/Users/pascal/Downloads/simple-object-tracking/models/bee_models/bee_object_detector.pb'

# path to the label map
PATH_TO_LABEL_MAP = 'data/label_map.pbtxt'

# PATH_TO_VIDEO = 'videos/spring_short.mov'
PATH_TO_VIDEO = 'videos/spring_2.mp4'

# PATH_TO_VIDEO = 'videos/bees_2.mp4'
#PATH_TO_VIDEO = 'videos/mucho_bees.mp4'

# number of classes
NUM_CLASSES = 1
THRESHOLD = 0.7
MASK = False

dateTimeObj = datetime.now()
time_stamp = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S.%f")

VIDEO = True

if VIDEO:
    cap = cv2.VideoCapture(PATH_TO_VIDEO)
else:
    cap = cv2.VideoCapture(0)

c.execute("select max(run_id) from od_runs")
run_id = c.fetchone()[0]
if not run_id:
    run_id = 1
else:
    run_id += 1


c.execute("""insert into od_runs values({},"{}",{},"{}")""".format(run_id,PATH_TO_FROZEN_GRAPH,THRESHOLD,time_stamp))

#subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)




ct = CentroidTracker()
trackableObjects = {}
trackers = []
fps = FPS().start()

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalIn = 0
totalOut = 0

image_center_x = 420
image_center_y = 350

# reads the frozen graph

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


outname = 'bee_output_{}.avi'.format(time_stamp)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(outname,fourcc,10.0,(int(width),int(height)))
detections = []
frame = 0

# Detection
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            frame += 1
            w_h = tf.constant([width,height,width,height], dtype=tf.float32)

            # Read frame from camera
            ret, image_np = cap.read()
            #image_np = subtractor.apply(image_np)
            if image_np is None:
                break
            rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np,axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detections
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            if MASK:
                cv2.circle(image_np, (image_center_x, image_center_y), 120, (0, 0, 0), -1)
                cv2.circle(image_np, (image_center_x, image_center_y), 600, (0, 0, 0), 400)
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            detections.append(int(num_detections[0]))
            #print("Boxes: {}, Length: {}".format(boxes, len(boxes[0])))
            coordinates = boxes[0] * w_h
            coordinates = coordinates.eval()

            coordinates = vis_util.return_coordinates(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=False,
                line_thickness=1,
                min_score_thresh=THRESHOLD)

            rects = []



            for i in range(len(coordinates)):
                ymin, ymax, xmin, xmax, conf = coordinates[i]
                if conf >= THRESHOLD:
                    rects.append([xmin,ymin,xmax,ymax])
                    startX = coordinates[i][0]
                    endX = coordinates[i][1]
                    startY = coordinates[i][2]
                    endY = coordinates[i][3]
                    X = (xmin + xmax) // 2
                    Y = (ymin + ymax) // 2

                    center = (X, Y)
                    print(center)

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)

                    c.execute("""insert into coordinates (run_id,video,frame,b_id,x_start,x_end,y_start, y_end,x_center,y_center,confidence) 
                    values ({},"{}",{},{},{},{},{},{},{},{},{})""".format(run_id,PATH_TO_VIDEO,frame,i,startX,endX,startY,endY,X,Y,conf))

            objects = ct.update(rects)
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                # text = "ID {}".format(i)
                # cv2.putText(image_np, text, (x - 10, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.circle(image_np, (x, y), 4, (0, 255, 0), -1)
                to = trackableObjects.get(objectID, None)
                text = "ID {}".format(objectID)
                cv2.circle(image_np, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid)
                # otherwise, there is a trackable object so we can utilize it
                # to determine direction
                else:
                    # the difference between the y-coordinate of the *current*
                    # centroid and the mean of *previous* centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')
                    y = [c[1] for c in to.centroids]
                    x = [c[0] for c in to.centroids]
                    print(x)
                    print(y)

                    direction_y = centroid[1] - np.mean(y)
                    direction_x = centroid[0] - np.mean(x)

                    print(direction_x,direction_y)
                    to.centroids.append(centroid)
                    distance_to_center = math.hypot(image_center_x-x[-1],image_center_y-y[-1])
                    text = "dist {}".format(int(distance_to_center))
                    cv2.arrowedLine(image_np, (int(np.mean(x)), int(np.mean(y))), (centroid[0], centroid[1]), (0, 255, 0),5)
                    # check to see if the object has been counted or not
                    if not to.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if direction_x < 0 and centroid[1] < height // 2:
                            totalIn += 1
                            to.counted = True
                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif direction_x > 0 and centroid[1] > height // 2:
                            totalOut += 1
                            to.counted = True

                # store the trackable object in our dictionary
                trackableObjects[objectID] = to

                cv2.putText(image_np, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)




            # print("Coordiates: {}".format(coordinates.eval()))
            #print("Scores: {}, Length: {}".format(scores, len(scores[0])))
            #print("Classes: {}, Length: {}".format(classes, len(classes[0])))
            #print("Num Detections: {}, Length: {}".format(num_detections, len(num_detections)))


            print("num detections: {}".format(int(len(coordinates))))
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=THRESHOLD,
                line_thickness=1
                )
            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("Nr of Bees", int(len(coordinates))),
                ("out", totalOut),
                ("in", totalIn)
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(image_np, text, (10, int(height) - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # cv2.putText(image_np, "Nr of Bees: {}".format(int(len(coordinates))), (10,20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(cv2.resize(image_np, (int(width), int(height))))

        # Display output
            cv2.imshow('',cv2.resize(image_np, (int(width), int(height))))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows
        conn.commit()
        conn.close()
        # stop the timer and display FPS information
        fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

print("Detections: {}".format(detections))
print("Max detected: {}".format(max(detections)))
