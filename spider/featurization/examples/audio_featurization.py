import os
import csv
import librosa
import numpy as np
import pandas as pd
from spider.featurization.audio_featurization import AudioFeaturization

# Read the test data csv
csv_file='data/testAudioData.csv'
df = pd.read_csv(csv_file)

# Read in the audio data specified by the csv
data = []
for idx, row in df.iterrows():
	filename = os.path.join('data/raw_data', row['filename'])
	datum, sampling_rate = librosa.load(filename)
	data.append(datum)

# Optional audio featurization parameter specification
frame_length = 0.050
overlap = 0.025

# Request feature generation
print("Generating features...")
featurizer = AudioFeaturization(
	sampling_rate=sampling_rate,
	frame_length=frame_length,
	overlap=overlap
)

features = featurizer.produce(data)

# Save features to disk
with open('features.csv', 'w+') as f:
	for feature in features:
		np.savetxt(f, feature)