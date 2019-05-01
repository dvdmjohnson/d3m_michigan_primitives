from d3m import container
from common_primitives.video_reader import VideoReaderPrimitive as VideoReader
from d3m.container.dataset import D3MDatasetLoader
import os
from common_primitives import dataset_to_dataframe
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

def _save_video(video, filename):
    output = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (video.shape[2], video.shape[1]))
    for frame in video:
        output.write(frame)
    output.release()


dataset_doc_path = os.path.join(os.path.abspath('../..'), 'tests/data/video_dataset_1/datasetDoc.json')
dataset = D3MDatasetLoader().load('file://'+dataset_doc_path)

df0 = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataset_to_dataframe.Hyperparams(dataframe_resource='0')).produce(inputs=dataset).value


vr = VideoReader()

input_df = vr.produce(inputs=df0).value

hp = TRECSHyperparams(default_alpha = 3.0, output_frames = 140, trecs_method = 'cvr')

trecs = TRECS(hyperparams=hp, random_seed=randint(0, 2**32-1))

output_vid = trecs.produce(inputs=input_df).value

#_save_video(output_vid[0], 'test.avi')

