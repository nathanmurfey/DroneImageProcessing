#!/usr/bin/env python
import cv2
import numpy
import sys
from time import time
import Saliency.frequency_tuned
import Saliency.signature
import Saliency.dictionary_frequency
from  Saliency.commons import minmaxnormalization


def main(img):
    saliency_methods = [ ('dictionary_ica_saliency', Saliency.dictionary_frequency.dictionary_saliency),
                         ('frequency_tuned', Saliency.frequency_tuned.frequency_tuned_saliency),
                         ('signature', Saliency.signature.signature_saliency),
                       ]

    for name, method in saliency_methods:
        print name
        t = time()
        sal_img = method(img.copy())
        t = t - time()
        sal_img = minmaxnormalization(sal_img)
        cv2.imshow('%s  took %ss'%(name, t),255 -  (255 * sal_img).astype('uint8'))
        cv2.imwrite(name+'.png',255 -  (255 * sal_img).astype('uint8'))
    cv2.waitKey()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img  = cv2.imread(sys.argv[1])
    else:
        cam = cv2.VideoCapture(0)
        status, img = cam.read()
    main(img)

