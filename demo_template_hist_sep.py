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
WIDTH = 1280
HEIGHT = 720

book = xlwt.Workbook(encoding="utf-8")

sheet1 = book.add_sheet("Sheet 1")

import matplotlib.image as mpimg

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('hist_sal.avi',fourcc, 25.0, (1280,720))

cam = cv2.VideoCapture('TestVideos/TestGroup2/flyover_two.h264')
display_frame =  np.zeros((HEIGHT,WIDTH), np.uint8)

object_list = []
object_index = 0

detections = 0
logging = 0

info_disp = np.zeros((360,640), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
fps = 0
start = time.time()
frame_index = 0
reduction = 4
counter = 0

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(121)
ay = fig.add_subplot(122)


while (cam.isOpened()):
    (flag, frame) = cam.read()
    if not flag:
        break
    #if (counter > 300):
    #    break
    if ((counter % 1) == 0):
        print counter

    # SALIENCY SELECTION --------------------------
        frame = frame[0:720, 0:1280]# Crop from x, y, w, h -> 100, 200, 300, 400

        #s


        #shrnk frame
        frame_dimension = (frame.shape[1]/reduction,frame.shape[0]/reduction)
        detection_frame = cv2.resize(frame, frame_dimension )
        #srs = SpectralResidualSaliency(detection_frame)
        #map = srs.get_saliency_map()
        #map = Saliency.dictionary_frequency.dictionary_saliency(frame)
        map = Saliency.signature.signature_saliency(detection_frame)
        #map = Saliency.boolean_map.compute_saliency(detection_frame)


        #map = cv2.GaussianBlur(map,(5,5),0)
        map = cv2.resize(map, (frame.shape[1], frame.shape[0]))
        sal_img = Saliency.commons.minmaxnormalization(map)
        sal_conv = 255 - (255 * sal_img).astype('uint8')
        #sal_conv = map.astype('uint8')

        #sal_conv = cv2.resize(sal_conv, (starting_frame_dimensions))

        # CONTOUR GRABBING ---------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(sal_conv, 150, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        end = time.time()
        frame_index = frame_index + 1

        duration = end - start

        if duration > 1:
            fps = frame_index
            frame_index = 0
            start = time.time()


        BUFFER = 60
        cv2.putText(info_disp, 'OBJ: '+ str(len(contours)), (30, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(info_disp, 'FPS: ' + str(fps), (30, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if len(contours) < 100:
            for cnt in contours:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                area = 3.141 * (radius ** 2)

                print area
                #if area > 30000 and area < 50000:
                #if area < 6000:            ## signature
                if area < 1300 and area > 50:
                #if True:
                    #cv2.circle(frame, center, radius, (255, 255, 255), 2)

                    x,y,w,h = cv2.boundingRect(cnt)
                    h = h + 40
                    w = w + 40
                    if (x > 20):
                        x = x - 20
                    if y > 10:
                        y = y - 20

                    object_detected = gray[y:y+h,x:x+w]
                    object_detected_c = frame[y:y + h, x:x + w]
                   # print object_detected.shape
                    if not object_detected.shape[0] == 0 and not object_detected.shape[1] == 0:
                        object_detected = cv2.resize(object_detected, (100,100))
                        object_list.insert(0,object_detected)
                        object_index = object_index + 1
                        #img = mpimg.imread(object_detected)

                        hist_full = cv2.calcHist([object_detected], [0], None, [256], [0, 256])
                        #if(0!=0):
                        #if True:
                        if(max(hist_full) > 1400):
                            cv2.circle(frame, center, radius, (255, 255, 255), 2)
                            #continue
                            #ax.clear()
                            #ay.clear()
                            #histr = cv2.calcHist([object_detected], [0], None, [256], [0, 256])
                            #ax.plot(hist_full)
                            #ax.set_ylim([0,2000])
                            #ax.xlim([0, 256])
                            #ay.imshow(object_detected, cmap='Greys_r')
                            #plt.show()
                            #ay.imshow(object_detected_c)
                            #fig.canvas.draw()
                            #fig.savefig('HistogramFigures/Report/ExperimentOne/hist_frame_' + str(counter) + '.png')
                            #if(max(hist_full) > 1400):
                               #fig.savefig('HistogramFigures/NewHist/pos/hist_frame_'+str(logging)+'.png')
                               #logging = logging + 1
                            #else:
                            #   fig.savefig('HistogramFigures/NewHist/neg/hist_frame_'+str(logging)+'.png')


                        #fig.savefig('HistogramFigures/hist_frame_'+str(counter)+'.png')

        frame = cv2.resize(frame, (640, 360))
        gray_circs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(gray, (640,360))
        sal_conv = cv2.resize(sal_conv, (640, 360))
        gray_circs = cv2.resize(gray_circs, (640,360))
        info_disp = cv2.resize(info_disp, (640, 360))

        display_frame[0:360,0:640] = gray
        display_frame[0:(360), 640:640+640] = sal_conv
        display_frame[360:(360+360),640:640 + 640] = info_disp
        display_frame[360:(360+360), 0:640] = gray_circs

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

book.save("test.xls")