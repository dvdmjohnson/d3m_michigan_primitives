from d3m import container
import os
import sys
import numpy as np
import cv2

import resource
import time
from random import randint
from shutil import copyfile
from spider.preprocessing.trecs import TRECS, TRECSHyperparams

'''
   Load a video from data/raw_data and resample it to "alpha" times it's original speed 
'''

def _load_video(path):
    """
    Loads a video from a given path

    Arguments:
        :path: Path to the video to load

    Returns:
        :vid: Video loaded from the specified location
    """

    cap = cv2.VideoCapture(path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        else:
            frames.append(frame)

    cap.release()

    return np.array(frames)




def _save_video(video, filename):
    output = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (video.shape[2], video.shape[1]))
    for frame in video:
        output.write(frame)
    output.release()

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/raw_data')

vid_path = os.path.join(data_path, 'v_Biking_g14_c03.avi' )
vid_path_list = container.List([vid_path, vid_path])

data = container.ndarray(_load_video(vid_path))
data_list = container.List([data, data])

hp = TRECSHyperparams(default_alpha = 3.0, output_frames = 140, trecs_method = 'cvr')

trecs = TRECS(hyperparams=hp, random_seed=randint(0, 2**32-1))

output_vid = trecs.produce(inputs=vid_path_list).value
output_vid = trecs.produce(inputs=data_list).value

#_save_video(output_vid[0], 'test.avi')



