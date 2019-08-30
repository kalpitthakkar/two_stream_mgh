import argparse
import os, glob
import numpy as np
import cv2

import multiprocessing
import threading
from datetime import datetime
from time import time
import tensorflow as tf

def saveOptFlowToImage(flow, basename, merge):
    if merge:
        # save x, y flows to r and g channels, since opencv reverses the colors
        cv2.imwrite(basename+'.png', flow[:,:,::-1])
    else:
        cv2.imwrite(basename+'_x.jpg', flow[...,0])
        cv2.imwrite(basename+'_y.jpg', flow[...,1])

def calc_opt_flow(vid_path, start, end, label):
    optflow = cv2.DualTVL1OpticalFlow_create()
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    if ret:
        prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for fnum in range(nframes):
        ret, frame = cap.read()
        if ret:  
            if start <= fnum and fnum <= end:
                curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = curr.shape
                flow = optflow.calc(prev, curr, None)
                #flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 7, 1.5, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                mag_thresh = np.mean(mag) + 3.0 * np.std(mag)
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr_noisy = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # Threshold the magnitude: Reject all values less than (mu + 3*std)
                mag[mag < mag_thresh] = 0
                flow[...,0], flow[...,1] = cv2.polarToCart(mag, ang, angleInDegrees=True)
                flow[...,0] = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
                flow[...,1] = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
                flow = np.concatenate((flow, np.zeros((h,w,1))), axis=2)
                hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imshow('prev', prev)
                cv2.imshow('curr', curr)
                cv2.imshow('opt_flow_bgr_noisy', bgr_noisy)
                cv2.imshow('opt_flow_bgr: ' + label, bgr)
                k = cv2.waitKey(250) & 0xff
                if k == 27:
                    break

                prev = curr
                #saveOptFlowToImage(flow, save_path, merge=True)
        else:
            break
    cap.release()
            
if __name__ == '__main__':
    tr_pickle_path = '/media/data_cifs/MGH/pickle_files/v2_selected/mgh_train.pkl'
    te_pickle_path = '/media/data_cifs/MGH/pickle_files/v2_selected/mgh_test.pkl'
    base_path = '/media/data_cifs/MGH/videos/'
    
    import pickle
    with open(tr_pickle_path, 'rb') as f:
        tr_out = pickle.load(f)
    with open(te_pickle_path, 'rb') as f:
        te_out = pickle.load(f)
   
    print("Checking flow output for action snippets in training ..")
    for val in tr_out:
        vname = val[0][0]
        s, e = val[0][1], val[0][2]
        lab = val[1]
        vid_path = os.path.join(base_path, vname)
        calc_opt_flow(vid_path, s, e, lab)
    
    print("Checking flow output for action snippets in testing ..")
    for val in te_out:
        vname = val[0][0]
        s, e = val[0][1], val[0][2]
        lab = val[1]
        vid_path = os.path.join(base_path, vname)
        calc_opt_flow(vid_path, s, e, lab)
