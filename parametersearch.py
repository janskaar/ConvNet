import tensorflow as tf
import os, shutil, time
import numpy as np
from single_run import Run

## Set up paths
test_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/300ms/test_data_ngj.tfrecord')
training_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/300ms/training_data_ngj.tfrecord')
validation_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/300ms/validation_data_ngj.tfrecord')
savepath = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/ConvNet/save/parametersearch/')


n_keep = 5
n_runs = 5

np.random.seed(1236754)

for param in ('eta', 'g', 'j'):
    if not os.path.isdir(os.path.join(savepath, param)):
        os.mkdir(os.path.join(savepath, param))
    for i in range(n_keep):
        if not os.path.isdir(os.path.join(savepath, param, str(i))):
            os.mkdir(os.path.join(savepath, param, str(i)))
        if not os.path.isdir(os.path.join(savepath, param, str(i), 'best')):
            os.mkdir(os.path.join(savepath, param, str(i), 'best'))
        if not os.path.isdir(os.path.join(savepath, param, str(i), 'workspace')):
            os.mkdir(os.path.join(savepath, param, str(i), 'workspace'))

reg_logrange = (2, 6)
lr_logrange = (4, 6)
batch_range = (50, 200)


for k, param in enumerate(('g', 'j')):
    param_index = k + 1
    with open(os.path.join(savepath, 'training_info.txt'), 'a') as f:
        f.write('TRAINING '+param+'\n')
    for i in range(n_runs):
        reg = 10**(-np.random.uniform(low=reg_logrange[0], high=reg_logrange[1]))
        lr = 10**(-np.random.uniform(low=lr_logrange[0], high=lr_logrange[1]))
        batch = int(np.random.uniform(low=batch_range[0], high=batch_range[1]))
        local_savepath = os.path.join(savepath, param, str(i))
        run = Run(training_path, test_path, validation_path, param_index, local_savepath, lr, reg, 1.0, save_step = 2, batch_size=batch)
        run.train()
        with open(os.path.join(savepath, 'training_info.txt'), 'a') as f:
            f.write('reg = %.6f, batch = %d, lr = %.6f, i=%d, best loss=%.6f \n'%(reg, batch, lr, i, run.best_loss))
        tf.reset_default_graph()
