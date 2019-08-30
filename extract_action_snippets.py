import numpy as np
import os, glob
import pickle
from tqdm import tqdm

import cv2

def extract_action_snippets(data_dir, entry, output_dir):
    vid_path = entry[0][0]
    vpath = os.path.join(data_dir, vid_path)
    if not os.path.exists(vpath):
        vpath = vpath[:-3] + 'avi'

    s, e = entry[0][1], entry[0][2]
    lab = entry[1]
    #if not '4c6cb293-aaae-4e47-bb1a-34b771393a41' in vpath:
    #    return
    print(vpath, s, e, lab)

    vid_dir = os.path.join(output_dir, vid_path.split('.')[0])
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)
    action_dir = os.path.join(vid_dir, lab, str(s)+'_'+str(e))
    if not os.path.exists(action_dir):
        os.makedirs(action_dir)

    cap = cv2.VideoCapture(vpath)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for fnum in range(nframes):
        save_path = os.path.join(action_dir, "img%06d.png" % (fnum))
        if os.path.exists(save_path):
            continue
        ret, frame = cap.read()
        if ret:
            if s <= fnum and fnum <= e:
                cv2.imwrite(os.path.join(action_dir, "img%06d.png" % (fnum)), frame)
            elif fnum > e:
                break
            else:
                continue
        else:
            break
    cap.release()

def snippets_from_metadata(metadata_path, data_dir, output_dir):
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    # Metadata structure: Each entry is of the form [(sub_vid_path, start, end), label]
    for i in tqdm(range(len(metadata))):
        entry = metadata[i]
        extract_action_snippets(data_dir, entry, output_dir)

if __name__ == '__main__':
    data_dir = "/media/data_cifs/MGH/videos"
    output_dir = "/media/data_cifs/MGH/v2_action_snippets"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #metadata_path = "/media/data_cifs/MGH/pickle_files/v2_selected/mgh_train.pkl"
    #snippets_from_metadata(metadata_path, data_dir, output_dir)
    metadata_path = "/media/data_cifs/MGH/pickle_files/v2_selected/mgh_test.pkl"
    snippets_from_metadata(metadata_path, data_dir, output_dir)
