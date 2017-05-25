import cv2
from Saliency import SpectralResidualSaliency
from Saliency import MR_saliency
from Saliency import IttiRapidSceneAnalysis
from Saliency import IttiSaliencyAttention
import Saliency.minimum_barrier_saliency
import Saliency.robust_background_detection
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

filepath = 'TestImages/30_m_sal_Test2.png'
test_image = cv2.imread(filepath)
test_image_clean = test_image.copy()
resize_factor = 4
print test_image.shape
test_image_copy = cv2.resize(test_image, (1280/resize_factor,720/resize_factor))
frame_count = 1
max_duration = 0
min_duration = 10000
total_duration = 0

mr = MR_saliency()
ir = IttiRapidSceneAnalysis(1280/resize_factor, 720/resize_factor)

for i in range(0,frame_count):
    print "FRAME: ", i, "  --  . ", test_image.shape
    start = time.time()

    ### -------------------------------  SALIENCY SELECTION --------------------------

    ## -- SPECTRAL RESIDUAL
    #srs = SpectralResidualSaliency(test_image)
    #map = srs.get_saliency_map()
    #sal_conv = (map * 255).round().astype(np.uint8)  # minimum barrier
    #cv2.imshow('frame', sal_conv)

    ## -- IMAGE SIGNATURE
    map = Saliency.signature.signature_saliency(test_image_copy)
    map = cv2.resize(map, (640,480));
    #sal_conv = 255 -  (255 * map).astype('uint8')
    sal_img = Saliency.commons.minmaxnormalization(map)
    #sal_conv = 255 - (255 * sal_img).astype('uint8')
    sal_conv = 255 - (255 * sal_img).astype('uint8')
    #cv2.imshow('frame', sal_conv)

    ## -- ITTI RAPID SCENE
    #map = ir.SMGetSM(test_image)
    #sal_conv = (map * 255).round().astype(np.uint8)
    #cv2.imshow('frame', sal_conv)

    ## -- ITTI SALIENCY ATTENTION
    #map = IttiSaliencyAttention(test_image).map
    #sal_img = Saliency.commons.minmaxnormalization(map)
    #sal_conv = (map * 255).round().astype(np.uint8)
    #cv2.imshow('frame', sal_conv)

    ## -- MINIMUM BARRIER SALIENCY
    #gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
    #map = Saliency.minimum_barrier_saliency.minimum_barrier_saliency(gray, 1)
    #sal_conv = (map * 255).round().astype(np.uint8)
    #sal_conv = cv2.resize(sal_conv, (1280, 720))
    #cv2.imshow('frame', sal_conv)

    ## -- ROBUST BACKGROUND  - COMPLETELY UNSUITABLE
    #map = Saliency.robust_background_detection.get_saliency_robust_background_detection(filepath)
    #sal_img = Saliency.commons.minmaxnormalization(map)
    #sal_conv = (map * 255).round().astype(np.uint8)
    #cv2.imshow('frame', sal_conv)

    ## -- MANIFOLD RANKINGS
    #map = mr.saliency(test_image)
    #sal_img = Saliency.commons.minmaxnormalization(map)
    #sal_conv = (map * 255).round().astype(np.uint8)
    #cv2.imshow('frame', sal_conv)

    ## -- BOOLEAN MAP
    #map = Saliency.boolean_map.compute_saliency(test_image)
    #sal_img = Saliency.commons.minmaxnormalization(map)
    #sal_conv = 255 - map
    #cv2.imshow('frame', sal_conv)

    ## -- DICTIONARY FREQUENCY
    #map = Saliency.dictionary_frequency.dictionary_saliency(test_image)
    #sal_img = Saliency.commons.minmaxnormalization(map)
    #sal_conv = 225-(sal_img * 255).round().astype(np.uint8)
    #sal_conv = cv2.resize(sal_conv, (1280,720))
    #cv2.imshow('frame', sal_conv)

    ## -- FREQUENCY TUNED
    #map = Saliency.frequency_tuned.frequency_tuned_saliency(test_image)
    #sal_img = Saliency.commons.minmaxnormalization(map)
    #sal_conv = (sal_img * 255).round().astype(np.uint8)
    #sal_conv = cv2.resize(sal_conv, (1280, 720))
    #cv2.imshow('frame', sal_conv)



    ### CONVERSION ------------------------------------
    ###  this is required for spectral residual
    #sal_conv = (map * 255).round().astype(np.uint8)  #minimum barrier
    ### this is required for the dictionary_frequency, signature and frequency tuned

    end = time.time()


    #sal_conv =(255 * sal_img).astype('uint8')
    #sal_conv = 255 -  (255 * sal_img).astype('uint8')
    #sal_conv = 255 - map # bool map
    #sal_conv =  map.astype('uint8')

    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(sal_conv, 248, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if True:
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            area = 3.141 * (radius ** 2)

            print area
            #if area > 30000 and area < 50000:
            #if area < 6000:            ## signature
            #if area < 10000:
            if True:
                cv2.circle(test_image_clean, center, radius, (255, 255, 255), 2)
            if area > 3000 and area < 6000:
                cv2.circle(test_image, center, radius, (255, 255, 255), 2)


        cv2.imshow('frame1', test_image)
        #cv2.imshow('frame2', sal_conv)
        #cv2.imshow('frame3', test_image_clearn)
        cv2.imshow('frame3', test_image_clean)
        cv2.imwrite('TestImages/Report/img_area_rej.png', test_image)
        #cv2.imwrite('TestImages/Report/img_sal.png', test_image_clean)
        #cv2.imwrite('TestImages/Report/img_clean.png', test_image_clearn)

    duration = end-start
    if duration > max_duration:
        max_duration = duration
    if duration < min_duration:
        min_duration = duration
    total_duration = total_duration + duration

average_duration = total_duration /frame_count

print "AVERAGE: ", average_duration
print "MIN: ", min_duration
print "MAX: ", max_duration
print "0.006"



#cv2.imshow('frame', sal_conv)
#sal_conv = cv2.resize(sal_conv, (1280, 720))
#cv2.imwrite('TestImages/rapidscene2.png', sal_conv)


while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
