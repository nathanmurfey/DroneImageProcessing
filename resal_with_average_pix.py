import cv2
from Saliency import SpectralResidualSaliency
import Saliency.dictionary_frequency
import Saliency.signature
import Saliency.boolean_map
import Saliency.frequency_tuned
import numpy as np
import time
from matplotlib import pyplot as plt
from datetime import datetime
import xlwt
import matplotlib.image as mpimg

def convert_to_hull(contours):
    hulls = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        hulls.insert(0,hull)
    return hulls

def get_saliency_frame(frame):
    frame = frame[0:HEIGHT, 0:WIDTH]  # Crop from x, y, w, h -> 100, 200, 300, 400
    frame_dimension = (frame.shape[1] / reduction, frame.shape[0] / reduction)
    detection_frame = cv2.resize(frame, frame_dimension)
    map = Saliency.signature.signature_saliency(detection_frame)
    map = cv2.resize(map, (frame.shape[1], frame.shape[0]))
    sal_img = Saliency.commons.minmaxnormalization(map)
    sal_conv = 255 - (255 * sal_img).astype('uint8')
    return sal_conv

def draw_detections(frame, contours):
    for cnt in contours:
        cv2.drawContours(frame, [cnt], 0, (0, 255, 100), 3)

def filter_by_area(contours):
    in_range_contours = []
    out_range_contours = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        area = 3.141 * (radius ** 2)

        if area < AREA_MAX and area > AREA_MIN:
            in_range_contours.insert(0, cnt)
        else:
            out_range_contours.insert(0, cnt)
    return in_range_contours, out_range_contours

def filter_by_histogram(gray, contours):
    new_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        h = h + 40
        w = w + 40
        if (x > 20):
            x = x - 20
        if y > 10:
            y = y - 20

        object_detected = gray[y:y + h, x:x + w]

        if not object_detected.shape[0] == 0 and not object_detected.shape[1] == 0:
            object_detected = cv2.resize(object_detected, (100,100))
            hist_full = cv2.calcHist([object_detected], [0], None, [256], [0, 256])
            if(max(hist_full) > 1400):
                new_contours.insert(0, cnt)
    return new_contours


# --- style
font = cv2.FONT_HERSHEY_SIMPLEX
WIDTH = 1280
HEIGHT = 720
AREA_MAX = 1300
AREA_MIN = 50
BUFFER = 60

# --- Output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('hist_sal.avi',fourcc, 25.0, (1280,720))

# --- Input
cam = cv2.VideoCapture('TestVideos/TestGroup2/flyover_two.h264')
display_frame =  np.zeros((HEIGHT,WIDTH), np.uint8)

object_list = []
object_index = 0

detections = 0

info_disp = np.zeros((360,640), np.uint8)

# --- FPS
fps = 0
start = time.time()
frame_index = 0

reduction = 4
counter = 0

while (cam.isOpened()):
    (flag, frame) = cam.read()

    # -- these are extra frames for the display
    pass_one_frame = frame.copy()
    pass_two_frame = frame.copy()
    pass_three_frame = frame.copy()

    fill_in_frame = frame.copy()

    if not flag:
        break
    if ((counter % 1) == 0):

        # --- BGR
        average_color_per_row = np.average(frame, axis=0)
        average_colors = np.average(average_color_per_row, axis=0)
        fill = np.full(frame.shape, average_colors, np.uint8)

        # --- Saliency
        sal_conv = get_saliency_frame(frame)

        # --- grab first contours
        ret, thresh = cv2.threshold(sal_conv, 150, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        end = time.time()
        frame_index = frame_index + 1

        duration = end - start

        if duration > 1:
            fps = frame_index
            frame_index = 0
            start = time.time()


        # get the normalisation
        # -- print to the info display
        cv2.putText(info_disp, 'OBJ: '+ str(len(contours)), (30, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(info_disp, 'FPS: ' + str(fps), (30, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # pass in the contours to draw and then filter the contours based on factors
        draw_detections(pass_one_frame, contours)
        contour_filter_one_in, contour_filter_one_out = filter_by_area(contours)
        draw_detections(pass_two_frame, contour_filter_one_in)
        contour_filter_two = filter_by_histogram(gray, contour_filter_one_in)
        draw_detections(pass_three_frame, contour_filter_two)

        ## lets try to get it better.

        mask = np.zeros(frame.shape, np.uint8)
        cv2.drawContours(mask, contour_filter_one_out, -1, (255, 255, 255), -1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(mask)

        # background
        fill_in_frame = cv2.bitwise_and(frame,frame,mask = mask)

        # foreground
        foreground_frame = np.zeros(frame.shape, np.uint8)
        foreground_frame = cv2.bitwise_and(fill, fill, mask=mask_inv)


        # new image
        fill_in_frame = cv2.add(fill_in_frame, foreground_frame)

        # sal again
        sal_conv_2 = get_saliency_frame(fill_in_frame)

        # DRAW DISPLAY

        frame = cv2.resize(frame, (640, 360))
        gray_circs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



        gray = cv2.resize(gray, (640,360))
        # 320, 180

        pass_one_frame = cv2.resize(pass_one_frame, (320,180))
        pass_two_frame = cv2.resize(pass_two_frame, (320,180))
        pass_three_frame = cv2.resize(pass_three_frame, (320, 180))
        pass_one_frame = cv2.cvtColor(pass_one_frame, cv2.COLOR_BGR2GRAY)
        pass_two_frame = cv2.cvtColor(pass_two_frame, cv2.COLOR_BGR2GRAY)
        pass_three_frame = cv2.cvtColor(pass_three_frame, cv2.COLOR_BGR2GRAY)
        cv2.putText(pass_one_frame,"All Contours", (1, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pass_two_frame, "By Area", (1, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pass_three_frame, "By Histogram", (1, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        fill = cv2.resize(fill, (320, 180))
        fill = cv2.cvtColor(fill, cv2.COLOR_BGR2GRAY)

        fill_in_frame = cv2.resize(fill_in_frame, (320, 180))
        fill_in_frame = cv2.cvtColor(fill_in_frame, cv2.COLOR_BGR2GRAY)

        #updated_frame = cv2.resize(updated_frame, (320, 180))
        #updated_frame = cv2.cvtColor(updated_frame, cv2.COLOR_BGR2GRAY)

        foreground_frame = cv2.resize(foreground_frame, (320, 180))
        foreground_frame = cv2.cvtColor(foreground_frame, cv2.COLOR_BGR2GRAY)

        sal_conv = cv2.resize(sal_conv, (320, 180))
        sal_conv_2 = cv2.resize(sal_conv_2, (320, 180))

        mask = cv2.resize(mask, (320,180))
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask_inv = cv2.resize(mask_inv, (320, 180))
        #mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_BGR2GRAY)

        gray_circs = cv2.resize(gray_circs, (320,180))
        info_disp = cv2.resize(info_disp, (320, 180))

        display_frame[0:360,0:640] = gray
        display_frame[0:(180), 640:640+320] = sal_conv
        display_frame[0:(180), 640+320:640+320+320] = mask
        display_frame[180:(180+180), 640:640 + 320] = mask_inv
        display_frame[180:(180+180), 640 + 320:640 + 320 + 320] = fill_in_frame
        display_frame[360:(360+180),640:640 + 320] = info_disp
        display_frame[360:(360 + 180), 640+320:640 + 320+320] = sal_conv_2
        display_frame[360:(360 + 180), 0:320] = pass_one_frame


        display_frame[360:(360 + 180), 320:320+320] = pass_two_frame
        display_frame[360+180:(360+180+180), 0:320] = pass_three_frame

        cv2.imshow("frame", display_frame)
        video_out = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
        out.write(video_out)


        print display_frame.shape


        if len(object_list) > 1:
            info_disp = np.zeros((360, 640), np.uint8)
            if not object_detected.shape[0] == 0 and not object_detected.shape[1] == 0:
                info_disp[200:300,30:130] = object_list.pop(0)
        elif len(object_list) == 1:
            new_info_disp = np.zeros((360, 640), np.uint8)
            new_info_disp[200:300, 30:130] = info_disp[200:300, 30:130]
            info_disp = new_info_disp
        else:
            info_disp = np.zeros((360, 640), np.uint8)

    counter = counter + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





out.release()
cam.release()
cv2.destroyAllWindows()
