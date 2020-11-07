import cv2
import numpy as np

PATH_TO_VIDEO = 'videos/mucho_bees.mp4'

cap = cv2.VideoCapture(PATH_TO_VIDEO)

subtractor = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=100, detectShadows=True)

while True:
    _,frame = cap.read()

    mask = subtractor.apply(frame)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break