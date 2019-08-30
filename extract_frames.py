import numpy as np
import os, glob
import pickle
from tqdm import tqdm

import cv2
import multiprocessing, threading
import tensorflow as tf

def extract_frames(tidx, ranges, videos, output_dir):
    sidx = ranges[tidx][0]
    eidx = ranges[tidx][1]
    
    for idx in range(sidx, eidx):
        vid_path = videos[idx]
        out_dir = os.path.join(output_dir, '/'.join(vid_path[:-4].split('/')[-2:]))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        cap = cv2.VideoCapture(vid_path)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fnum in range(nframes):
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(out_dir, "img%06d.png" % (fnum)), frame)
            else:
                break
        cap.release()
        print("[Thread {}]: Finished extracting frames from video {}".format(tidx, os.path.basename(vid_path)))

if __name__ == '__main__':
    data_dir = "/media/data_cifs/MGH/videos"
    output_dir = "/media/data_cifs/MGH/video_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_threads = int(multiprocessing.cpu_count() / 2)
    ranges = []
    threads = []

    videos = glob.glob(os.path.join(data_dir, '*', '*.avi'))
    videos += glob.glob(os.path.join(data_dir, '*', '*.mp4'))
    spacings = np.linspace(0, len(videos), num_threads+1).astype(np.int)
    for i in range(len(spacings)-1):
        ranges.append((spacings[i], spacings[i+1]))

    print("Using {} spacings: {}".format(num_threads, ranges))

    coord = tf.train.Coordinator()
    for tidx in range(len(ranges)):
        args = (tidx, ranges, videos, output_dir)
        t = threading.Thread(
            target=extract_frames,
            args=args
        )
        t.start()
        threads.append(t)
    coord.join(threads)
