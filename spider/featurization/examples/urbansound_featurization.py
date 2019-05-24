"""
Achieves 71.3 % accuracy with a one-vs-rest SVM on urbansound with current
hyperparameters and no aggregation
"""

import argparse
import librosa
import numpy as np
import os
import pandas as pd
import random
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from spider.featurization.audio_featurization import AudioFeaturization
    
def featurize(input_path, output_path, audio_dir):

    # Load data from csv
    train_csv = os.path.join(input_path, 'trainData.csv')
    target_csv = os.path.join(input_path, 'trainTargets.csv')
    df = pd.read_csv(train_csv)
    df = df.assign(label=pd.read_csv(target_csv)['class'])

    # Deterministic numeric labeling
    all_labels = df['label'].unique()
    all_labels.sort()
    label_dict = dict(list(zip(all_labels, np.arange(len(all_labels)))))

    # Load and featurize audio
    for idx, row in df.iterrows():
        filename = os.path.join(input_path, audio_dir, row['filename'])
        audio_clip, sampling_rate = librosa.load(filename, sr=None)
        start = int(sampling_rate * float(row['start']))
        end = int(sampling_rate * float(row['end']))
        featurizer = AudioFeaturization(
            sampling_rate=sampling_rate,
            frame_length=0.20,
            overlap=0.10
        )
        features = featurizer.produce([audio_clip[start:end]])
        label = label_dict[row['label']]

        if features[1].size == 0:
            continue

        # Save features and labels to disk in numpy file
        output_name = os.path.join(output_path, str(row['d3mIndex']))
        np.savez(output_name, features=features[1], label=label)

def svm_classification(output_path):

    # 80-20 train-validation split
    files = os.listdir(output_path)
    random.shuffle(files)
    split_point = int(0.8*len(files))
    train_files = files[:split_point]
    valid_files = files[split_point:]

    # Load features and labels from disk
    trainX, trainY = load_features(output_path, train_files)
    validX, validY = load_features(output_path, valid_files)

    # Run a one-vs-rest svm classification
    classifier = svm.SVC(verbose=True, decision_function_shape='ovr')
    classifier.fit(trainX, trainY)
    print("Classification accuracy is ", accuracy_score(
        validY, classifier.predict(validX)
    ))

def load_features(dir, file_list):

    # Load all features from list into a numpy array
    features = []
    labels = []
    for file in file_list:
        npzfile = np.load(os.path.join(dir, file))
        for feature in npzfile['features']:
            features.append(feature)
            labels.append(npzfile['label'])

    return np.array(features), np.array(labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True)
    parser.add_argument('-o', '--output_path', required=True)
    parser.add_argument('-a', '--audio_dir', default='raw_audio')
    parser.add_argument('-f', '--featurize', default='1')
    parser.add_argument('-c', '--classify', default='1')
    args = parser.parse_args()
    if bool(int(args.featurize)):
        featurize(args.input_path, args.output_path, args.audio_dir)
    if bool(int(args.classify)):
        svm_classification(args.output_path)