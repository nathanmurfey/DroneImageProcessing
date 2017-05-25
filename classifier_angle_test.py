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
padding_px = 0

image = cv2.imread('TestImages/TestGroup1/easy_class1.jpg')
image = image[padding_px:image.shape[0]-padding_px,padding_px:image.shape[1]-padding_px]
test_image = cv2.resize(image, (1280,720))
resize_factor = 1

#test_image = cv2.resize(test_image, (1280/resize_factor,720/resize_factor))
frame_count = 1
max_duration = 0
min_duration = 10000
total_duration = 0

haar_classifier = HAARClassifier()
hog_classifer = HOGClassifier()

for i in range(1,19):
    print "FRAME: ", i
    start = time.time()
    if i != 7:
        image = cv2.imread('TestImages/TestGroup1/easy_class'+str(i)+'.jpg')
    else:
        image = cv2.imread('TestImages/TestGroup1/easy_class1.jpg')
    image = image[padding_px:image.shape[0] - padding_px, padding_px + 500:image.shape[1] - padding_px - 400]
    test_image = cv2.resize(image, (1280, 720))
    resize_factor = 1

    rects, weights = hog_classifer.detect(test_image)
    bodies = haar_classifier.detect_body(test_image)
    upper = haar_classifier.detect_upper(test_image)
    lower = haar_classifier.detect_lower(test_image)

    for (x, y, w, h) in rects:
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        object = test_image[y:y+h, x:x+w, :]
        object = cv2.resize(object, (100,200))
        faces = haar_classifier.detect_faces(object)


    for (x, y, w, h) in bodies:
        area = h * w
        if area > 15000:
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    for (x, y, w, h) in upper:
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in lower:
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 0, 255), 2)


    end = time.time()
    duration = end-start
    if duration > max_duration:
        max_duration = duration
    if duration < min_duration:
        min_duration = duration
    total_duration = total_duration + duration

    cv2.imshow('frame_c_'+str(i)+'', test_image)
    cv2.imwrite('TestImages/Appendix/image_ce_'+str(i)+'.jpg', test_image)


average_duration = total_duration /frame_count

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
