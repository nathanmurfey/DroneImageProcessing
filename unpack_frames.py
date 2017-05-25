import cv2
from Saliency import SpectralResidualSaliency
import Saliency.dictionary_frequency
import numpy as np
import time
from datetime import datetime
WIDTH = 1280
HEIGHT = 720

cam = cv2.VideoCapture('TestVideos/TestGroup2/hover_with_people_one.h264')

i = 0
while (cam.isOpened()):
    (flag, frame) = cam.read()
    if not flag:
        break
    filename = 'TestImages/TestGroup2/frame_r_'+str(i)+'.png'
    frame = frame[0:720, 0:1280]  # Crop from x, y, w, h -> 100, 200, 300, 400
    cv2.imwrite(filename,frame)
    print filename, "::", frame.shape[0], "--" ,frame.shape[1]
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    i = i + 1

cam.release()
cv2.destroyAllWindows()
