import cv2
from Classifiers import HOGClassifier
from Classifiers import HAARClassifier
from Saliency import SpectralResidualSaliency
import Saliency.dictionary_frequency
import Saliency.signature
import Saliency.boolean_map
import Saliency.frequency_tuned
import numpy as np
import time
from datetime import datetime
WIDTH = 1280
HEIGHT = 720


cam = cv2.VideoCapture('demo_footage_multi.mp4')
display_frame =  np.zeros((HEIGHT,WIDTH), np.uint8)

object_list = []
object_index = 0

detections = 0

hog = HOGClassifier()
haar = HAARClassifier()

info_disp = np.zeros((360,640), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
fps = 0
start = time.time()
frame_index = 0
frame_counter = 0
reduction = 6

while (cam.isOpened()):
    (flag, frame) = cam.read()
    frame = frame[0:720, 0:1280]
    if not flag:
        break
    if frame_counter % 10 != 0:
        frame_counter = frame_counter + 1
        continue

    #contours, w = hog.detect(frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    contours = haar.detect_body(frame_gray)
    faces = haar.detect_faces(frame_gray)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    end = time.time()
    frame_index = frame_index + 1
    frame_counter = frame_counter + 1
    print frame_counter


    duration = end - start

    if duration > 1:
        fps = frame_index
        frame_index = 0
        start = time.time()


    BUFFER = 60
    cv2.putText(info_disp, 'OBJ: '+ str(len(contours)), (30, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(info_disp, 'FPS: ' + str(fps), (30, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if len(contours) != 0:
        for (x, y, w, h) in contours:
            print "BOOM in the frame ::", frame_counter
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('window', frame)
            #(x, y), radius = cv2.minEnclosingCircle(cnt)
            #center = (int(x), int(y))
            #radius = int(radius)
            #area = 3.141 * (radius ** 2)

            #print area
            #if area > 30000 and area < 50000:
            #if area < 6000:            ## signature




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cam.release()
cv2.destroyAllWindows()

