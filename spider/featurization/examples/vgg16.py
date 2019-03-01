import os
import sys
import csv
import numpy as np

import resource
import time
from shutil import copyfile
from spider.featurization.vgg16 import VGG16

'''
    Generates features from images in R-22 dataset, then saves into features.csv file
'''

# num_samples: [subset r_22, r_22, Imagenet(animals, people, plants, artifacts)]
# [5, 225, 5337]

# disable TF warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/testData.csv')
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/raw_data')

# read csv file data
with open(csv_path, 'rb') as data_file:
    data_list = list(csv.reader(data_file, delimiter=','))
    # chop off column headings
    del data_list[0]

# row of image filenames
image_files = [row[1] for row in data_list]

# time and track memeory usage of primitive
start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
start_time = time.time()

VGG16 = VGG16()

images = []
print("Generating features...")
for image in image_files:
    images.append(os.path.join(data_path, image))

features = VGG16.produce(images)

end_time = time.time()
end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(("Time:\t" + str(end_time - start_time) + " sec")) # Time in seconds
print(("Memory:\t" + str((end_mem - start_mem)/1000) + " KB")) # CPU memory in KB
