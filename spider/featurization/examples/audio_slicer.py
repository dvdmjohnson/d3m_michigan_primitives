import librosa
import numpy as np
import os
import pandas as pd
from spider.featurization.audio_slicer import AudioSlicer

# Read the test data csv
csv_file='data/testAudioData.csv'
df = pd.read_csv(csv_file)

# Read in the audio data specified by the csv
data = []
for idx, row in df.iterrows():
	filename = os.path.join('data/raw_data', row['filename'])
	datum, sampling_rate = librosa.load(filename)
	data.append(datum)

frame_length = 1.0 # Sliced audio length in seconds
overlap = 0.5      # Overlap in seconds of adjacent clips
pad = True         # Pad clips shorter than frame_length

featurizer = AudioSlicer(sampling_rate, frame_length, overlap, pad)
features = featurizer.produce(data)

np.savetxt('features.csv', features[0])