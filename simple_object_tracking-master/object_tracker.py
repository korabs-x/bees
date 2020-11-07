# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# Tensorflow part


# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from collections import deque


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
				help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
				help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.75,
				help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video",
				help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=5,
				help="max buffer size")
args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

pts = deque(maxlen=args['buffer'])
pts_lst = [deque(maxlen=args['buffer']) for i in range(0,5)]


# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")





# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)



# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	#ret,frame = cap.read()

	if frame is None:
		break

	frame = imutils.resize(frame, width=300)


	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	print(detections)
	rects = []



	print(len(pts_lst))

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold

		if detections[0, 0, i, 2] > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
			x = (endX+startX)//2
			y = (endY+startY)//2
			center = (x,y)
			pts_lst[i].appendleft(center)
			print(pts_lst)



	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)
	for n in range(0, len(pts_lst)):
		for i in range(1, len(pts_lst[n])):
			# loop over the tracked objects
			for (objectID, centroid) in objects.items():


				# if either of the tracked points are None, ignore them
				if pts_lst[n][i - 1] is None or pts_lst[n][i] is None:
					continue
				# otherwise, compute the thickness of the line and
				# draw the connecting lines
				thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1)
				cv2.line(frame, pts_lst[n][i - 1], pts_lst[n][i], (0, 255, 0), thickness)

				# draw both the ID of the object and the centroid of the
				# object on the output frame
				text = "{}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 1, centroid[1] - 0),
					cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
				cv2.circle(frame, (centroid[0], centroid[1]), 1, (0, 255, 0), -1)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()