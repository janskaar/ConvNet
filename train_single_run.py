import tensorflow as tf
import os, shutil, time
import numpy as np
from single_run import Run

## Set up paths
test_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/300ms/test_data_ngj_augmented.tfrecord')
training_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/300ms/training_data_ngj_augmented.tfrecord')
validation_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/300ms/validation_data_ngj_augmented.tfrecord')
savepath = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/ConvNet/save/parametersearch/')


params = ('eta', 'g', 'j')
param_index = 2             # 0 for eta, 1 for g, 2 for j
lr = 1e-4
reg = 0.
batch = 300
load = False
softmax = False
keep_prob = 1.0
decay = 0.97
decay_step = 200
momentum = 0.9



path = os.path.join(savepath, params[param_index])
run = Run(training_path, test_path, validation_path, param_index, path, lr, reg, keep_prob,
          save_step = 1, batch_size=batch, load=load, softmax=softmax, decay=decay, decay_step=decay_step)
run.train()
tf.reset_default_graph()
