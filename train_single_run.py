import tensorflow as tf
import os, shutil, time
import numpy as np
from single_run import Run

## Set up paths
test_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/300ms/test_data_ngj.tfrecord')
training_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/300ms/training_data_ngj.tfrecord')
validation_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/300ms/validation_data_ngj.tfrecord')
savepath = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/ConvNet/save/parametersearch/g/')

lr = 1e-5
reg = 0
param_index = 1
batch = 50
load=False
softmax=False

run = Run(training_path, test_path, validation_path, param_index, savepath, lr, reg, 1.0, save_step = 2, batch_size=batch, load=load, softmax=softmax)
run.train()
