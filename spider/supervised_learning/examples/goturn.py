import os
import sys
from random import randint
import numpy as np

from d3m import container
from d3m.container.dataset import D3MDatasetLoader
import common_primitives 
from common_primitives import dataset_to_dataframe
from common_primitives import video_reader
import resource
import time
from shutil import copyfile
from spider.supervised_learning.goturn import GoTurn, GoTurnHyperparams

#time and track memory usage of primitive
start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
start_time = time.time()

hp = GoTurnHyperparams(num_gpus=1, num_epochs=1, learning_rate=1e-5, momentum=0.9, weight_decay=0.0005)
goturn = GoTurn(hyperparams=hp, random_seed=randint(0,2**32-1))

parent_directory = os.path.abspath(os.path.join(os.path.dirname(common_primitives.__file__),os.pardir))
dataset_doc_path = os.path.join(parent_directory,'tests','data','datasets','video_dataset_1','datasetDoc.json')
dataset = D3MDatasetLoader().load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

df0 =  dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataset_to_dataframe.Hyperparams(dataframe_resource='0')).produce(inputs=dataset).value

vr_hyperparams_class = video_reader.VideoReaderPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
vr = video_reader.VideoReaderPrimitive(hyperparams=vr_hyperparams_class.defaults())
input_df = vr.produce(inputs=df0).value

#Train
train_targets = container.ndarray(np.random.rand(len(input_df.iloc[:,1][0]),4)) #Randomize targets for testing
goturn.set_training_data(inputs=input_df, outputs=train_targets)
goturn.fit()

#Test
predictions = goturn.produce(inputs=input_df, targets=train_targets).value

end_time = time.time()
end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(("Time:\t" + str(end_time - start_time) + " sec")) #Time in seconds
print(("Memory:\t" + str((end_mem - start_mem)/1000) + " KB")) #CPU memory in KB
