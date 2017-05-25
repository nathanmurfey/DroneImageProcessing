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

display = np.zeros((720,1280, 3), np.uint8)
padding_px = 800
image = cv2.imread('TestImages/TestGroup1/hard_class2.jpg')
image = image[padding_px:image.shape[0]-padding_px,padding_px+500:image.shape[1]-padding_px-400]
test_image = cv2.resize(image, (1280,720))
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
    bodies = haar_classifier.detect_body(test_image)
    upper = haar_classifier.detect_upper(test_image)
    lower = haar_classifier.detect_lower(test_image)

    print upper


    for (x, y, w, h) in rects:
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        object = test_image[y:y+h, x:x+w, :]
        print object.shape
        object = cv2.resize(object, (100,200))
        cv2.imshow('object', object)
        top_down = 50
        across = 100
        display[top_down:top_down+200,1000+across:1000+ 100+across,:] = object
        faces = haar_classifier.detect_faces(object)
        print faces, "--<"

        cv2.putText(display, "HOG :" + str(round(weights[0][0], 3)), (1000+5, top_down+object.shape[0]+60), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    for (x, y, w, h) in bodies:
        area = h * w

        if area > 15000:
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(display, "HRB :1.0/" + str(round(len(bodies), 1)), (1000 + 5, top_down + object.shape[0] + 60+30), font, 1,(255, 255, 255), 1, cv2.LINE_AA)

    for (x, y, w, h) in upper:
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.putText(display, "HRU :1.0/" + str(round(len(upper), 1)), (1000 + 5, top_down + object.shape[0] + 60 + 30 + 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    for (x, y, w, h) in lower:
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(display, "HRL :1.0/" + str(round(len(lower), 1)), (1000 + 5, top_down + object.shape[0] + 60 + 30 + 30 + 30), font, 1,(255, 255, 255), 1, cv2.LINE_AA)


    end = time.time()
    duration = end-start
    if duration > max_duration:
        max_duration = duration
    if duration < min_duration:
        min_duration = duration
    total_duration = total_duration + duration

    cv2.putText(display, "CMT :" + str(round(duration, 2)),(1000 + 5, top_down + object.shape[0] + 60 + 30 + 30 + 30+30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

average_duration = total_duration /frame_count


#hog_classifer.run_scale_bench(test_image)

print "AVERAGE: ", average_duration
print "MIN: ", min_duration
print "MAX: ", max_duration
print "0.006"


print test_image.shape
print display.shape

smaller_image = test_image[0:720,0:1000,:]
display[0:720,0:1000, :] = smaller_image

cv2.imshow('frame2', display)
cv2.imwrite('TestImages/classifier_demo.png', display)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
