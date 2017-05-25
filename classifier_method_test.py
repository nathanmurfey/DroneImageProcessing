import cv2
from Classifiers import HOGClassifier
from Classifiers import HAARClassifier
from Saliency import SpectralResidualSaliency
from Saliency import MR_saliency
from Saliency import IttiRapidSceneAnalysis
import Saliency.dictionary_frequency
import Saliency.frequency_tuned
import Saliency.signature
import Saliency.boolean_map
import Saliency.commons
import numpy as np
import time
from datetime import datetime
WIDTH = 1280
HEIGHT = 720

font = cv2.FONT_HERSHEY_SIMPLEX

test_image = cv2.resize(cv2.imread('TestImages/TestGroup1/easy_class1.png'), (1280,720))

resize_factor = 1

#test_image = cv2.resize(test_image, (1280/resize_factor,720/resize_factor))
frame_count = 1
max_duration = 0
min_duration = 10000
total_duration = 0

haar_classifier = HAARClassifier()
hog_classifer = HOGClassifier()

for i in range(0,frame_count):
    print "FRAME: ", i
    start = time.time()


    rects, weights = hog_classifer.detect(test_image)

    downscaled_image = cv2.resize(test_image, (1280/4,720/4))
    map = Saliency.dictionary_frequency.dictionary_saliency(downscaled_image)
    sal_img = Saliency.commons.minmaxnormalization(map)
    sal_conv = 255-(sal_img * 255).round().astype(np.uint8)
    sal_conv = cv2.resize(sal_conv, (1280, 720))
    #ssal_conv = cv2.GaussianBlur(sal_conv,(9,9),0)

    #sal_conv = cv2.equalizeHist(sal_conv)
    gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

    print sal_conv[0,0]
    print gray[0,0]

    ret, thresh = cv2.threshold(sal_conv, 100, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('frame3', thresh)

    cv2.imshow('frame', sal_conv)

    for (x, y, w, h) in rects:
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #for (x, y, w, h) in bodies:
    #    cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    end = time.time()
    duration = end-start
    if duration > max_duration:
        max_duration = duration
    if duration < min_duration:
        min_duration = duration
    total_duration = total_duration + duration

average_duration = total_duration /frame_count

#hog_classifer.run_scale_bench(test_image)

print "AVERAGE: ", average_duration
print "MIN: ", min_duration
print "MAX: ", max_duration
print "0.006"

#cv2.imshow('frame2', test_image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
