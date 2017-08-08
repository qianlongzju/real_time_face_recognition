# coding: utf-8
import tensorflow as tf
import numpy as np
import cv2
import os
from os.path import join as pjoin
import sys
import copy
import random
import time
from skimage import exposure
import detect_face

#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

gpu_memory_fraction = 0.5
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def read_img(f):
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img

def face_detection(img):
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0] #number of faces
    return nrof_faces

def main():
    tStart = time.time()
    i = 0
    directory = sys.argv[1]
    for f in os.listdir(directory):
        #print f
        i += 1
        abs_f = pjoin(directory, f)
        img = read_img(abs_f)
        nrof_faces = face_detection(img)
        #nrof_faces = adjust_exposure(img)
        if nrof_faces == 0:
            print f
        #if nrof_faces == 0:
        #    cv2.imwrite('./result/false_20170807/%s.jpg' % f, img)
        #else:
        #    cv2.imwrite('./result/true_20170807/%s.jpg' % f, img)
        #break
    tEnd = time.time()
    print "It cost %f secs to detect %s images" % ((tEnd - tStart), i)
    print "it can detect %s images per second" % (i / (tEnd - tStart))

def adjust_exposure(img):
    gammas = [0.1 * i for i in range(5, 15)][::-1]
    nrof = []
    for gamma in gammas:
        image = exposure.adjust_gamma(img, gamma)
        nrof.append(face_detection(image))
    return max(nrof)

if __name__ == '__main__':
    main()
    #adjust_exposure(sys.argv[1])
