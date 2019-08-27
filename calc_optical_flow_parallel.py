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

def calc_opt_flow(thread_index, ranges, vpaths, output_dirs, start_times, curr_times, debug=False):
    start_idx = ranges[thread_index][0]
    end_idx = ranges[thread_index][1]

    start_times[thread_index] = time()
    for idx in range(start_idx, end_idx+1):
        curr_times[thread_index] = time()
        video_path = vpaths[idx]
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        hsv = np.zeros_like(frame)
        hsv[...,1] = 255
        
        save_dir = os.path.join(output_dirs[idx], os.path.basename(video_path).split('.')[0])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for fnum in range(nframes):
            ret, frame = cap.read()
            if ret:
                curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = curr.shape
                optflow = cv2.DualTVL1OpticalFlow_create()
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
                if debug:
                    cv2.imshow('prev', prev)
                    cv2.imshow('curr', curr)
                    cv2.imshow('opt_flow_bgr_noisy', bgr_noisy)
                    cv2.imshow('opt_flow_bgr', bgr)
                    k = cv2.waitKey() & 0xff
                    if k == 27:
                        break

                prev = curr
                saveOptFlowToImage(flow, os.path.join(save_dir, "frame%06d" % (fnum)), merge=True)
                
                if ((fnum > 0) and (fnum % 50 == 0)):
                    tm = time()
                    print("[{}][Thread {}]: Processed frames {} to {} for video {}".format(datetime.now().strftime("%Y-%m-%d %H:%M"), thread_index, fnum-50, fnum-1, os.path.basename(video_path)))
                    print("[Thread {}]: Time taken for 50 frames is {} seconds ({} FPS)".format(thread_index, round(tm-curr_times[thread_index], 3), round(50.0/(tm-curr_times[thread_index]), 2)))
                    curr_times[thread_index] = time()
            else:
                break

        tm = time()
        print("[{}][Thread {}]: Completed processing video {}".format(datetime.now().strftime("%Y-%m-%d %H:%M"), thread_index, os.path.basename(video_path)))
        print("[Thread {}]: Time taken for video is {} seconds".format(round(tm-start_times[thread_index], 3)))
        start_times[thread_index] = time()

if __name__ == '__main__':
    subs = ['BW46', 'MG51b', 'MG117', 'MG118', 'MG120b']
    output_dir = "/media/data_cifs/MGH/optical_flow"
    data_dir = "/media/data_cifs/MGH/videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    video_paths = []
    output_dirs = []
    for j in range(len(subs)):
        sub = subs[j]
        sub_dir = os.path.join(output_dir, sub)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        video_paths += glob.glob(os.path.join(data_dir, sub, '*.avi'))
        video_paths += glob.glob(os.path.join(data_dir, sub, '*.mp4'))
        for v in video_paths:
            output_dirs.append(os.path.join(sub_dir))

    num_threads = multiprocessing.cpu_count()
    spacings = np.linspace(0, len(video_paths), num_threads+1).astype(np.int)
    ranges = []
    threads = []
    start_times = [0] * num_threads
    curr_times = [0] * num_threads

    for i in range(len(spacings)-1):
        ranges.append((spacings[i], spacings[i+1]))

    print("Created {} chunks with spacings: {}".format(num_threads, ranges))

    coord = tf.train.Coordinator()
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges,
            video_paths, output_dirs, start_times, curr_times)

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
