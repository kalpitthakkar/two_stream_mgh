import argparse
import os, glob
import numpy as np
import cv2

import multiprocessing
import threading
from datetime import datetime
from time import perf_counter
import tensorflow as tf

def saveOptFlowToImage(flow, basename, merge):
    if merge:
        # save x, y flows to r and g channels, since opencv reverses the colors
        cv2.imwrite(basename+'.png', flow[:,:,::-1])
    else:
        cv2.imwrite(basename+'_x.jpg', flow[...,0])
        cv2.imwrite(basename+'_y.jpg', flow[...,1])

def calc_opt_flow(thread_index, ranges, starts, ends, vid_dirs, output_dir, debug=False):
    start_idx = ranges[thread_index][0]
    end_idx = ranges[thread_index][1]

    optflow = cv2.DualTVL1OpticalFlow_create()
    for idx in range(start_idx, end_idx+1):
        starts[thread_index] = perf_counter()
        print("[Thread {}] {}".format(thread_index, vid_dirs[idx]))
        imgs = sorted(glob.glob(os.path.join(vid_dirs[idx], '*.png')))
        
        prev = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2GRAY)
        
        #hsv = np.zeros_like(frame)
        #hsv[...,1] = 255
       
        suffix = '/'.join(vid_dirs[idx].split('/')[5:])
        save_dir = os.path.join(output_dir, suffix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(1, len(imgs)):
            fname = os.path.basename(imgs[i])
            save_path = os.path.join(save_dir, "flow_" + fname.replace('png', ''))
            if os.path.exists(save_path):
                continue
            curr = cv2.cvtColor(cv2.imread(imgs[i]), cv2.COLOR_BGR2GRAY)
            h, w = curr.shape
            flow = optflow.calc(prev, curr, None)
            #flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 7, 1.5, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mag_thresh = np.mean(mag) + 3.0 * np.std(mag)
            #hsv[...,0] = ang*180/np.pi/2
            #hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            #bgr_noisy = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # Threshold the magnitude: Reject all values less than (mu + 3*std)
            mag[mag < mag_thresh] = 0
            flow[...,0], flow[...,1] = cv2.polarToCart(mag, ang, angleInDegrees=True)
            flow[...,0] = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
            flow[...,1] = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
            flow = np.concatenate((flow, np.zeros((h,w,1))), axis=2)
            #hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            #bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            #if debug:
            #    cv2.imshow('prev', prev)
            #    cv2.imshow('curr', curr)
            #    cv2.imshow('opt_flow_bgr_noisy', bgr_noisy)
            #    cv2.imshow('opt_flow_bgr', bgr)
            #    k = cv2.waitKey() & 0xff
            #    if k == 27:
            #        break

            prev = curr
            saveOptFlowToImage(flow, save_path, merge=True)
            
        ends[thread_index] = perf_counter()
        t = ends[thread_index] - starts[thread_index]
        print("[{}][Thread {}]: Time taken for snippet {} is {} seconds [{} FPS]".format(datetime.now().strftime("%Y-%m-%d %H:%M"), thread_index, vid_dirs[idx], round(t, 3), round(len(imgs)/t, 3)))

if __name__ == '__main__':
    subs = ['BW46', 'MG51b', 'MG117', 'MG118', 'MG120b']
    output_dir = "/media/data_cifs/MGH/optical_flow"
    data_dir = "/media/data_cifs/MGH/v2_action_snippets"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    video_dirs = glob.glob(os.path.join(data_dir, '*', '*', '*', '*'))
    print(len(video_dirs))

    num_threads = int(multiprocessing.cpu_count())
    spacings = np.linspace(0, len(video_dirs), num_threads+1).astype(np.int)
    ranges = []
    threads = []
    starts = [0] * num_threads
    ends = [0] * num_threads

    for i in range(len(spacings)-1):
        ranges.append((spacings[i], spacings[i+1]))

    print("Created {} chunks with spacings: {}".format(num_threads, ranges))

    coord = tf.train.Coordinator()
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, starts, ends, video_dirs, output_dir)

        t = threading.Thread(
            target=calc_opt_flow,
            args=args
        )
        t.start()
        threads.append(t)
    coord.join(threads)

    #vid_path = "/media/data_cifs/MGH/videos/MG117/Xxxxxxx~ Xxxxx_2276c0a7-62b5-4162-aec4-ead76126bdad_0045.avi"
    #vid_path = "/media/data_cifs/MGH/videos/MG51b/Xxxxxx~ Xxxxxx_ceb2b945-a1b1-4627-8155-7c73937fcdb9_0158.avi"
    #calc_opt_flow(video_path=vid_path, debug=False)
