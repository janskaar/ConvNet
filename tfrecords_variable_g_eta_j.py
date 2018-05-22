import os
import numpy as np
import tensorflow as tf
from ConvNet.hdf5_to_tfrecords import h5_to_slices, create_tfrecords
import random

top_dir = os.path.join('C:\\Users\\JanEirik\\Documents\\TensorFlow\\data\\hybrid_brunel47\\nest_output')
output_path = os.path.join('C:\\Users\\JanEirik\\Documents\\TensorFlow\\tfrecords_data\\final\\900ms\\')

etas = np.array([1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0], dtype=np.float32)
gs = np.array([4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4], dtype=np.float32)
js = np.array([0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2], dtype=np.float32)

def read_ids_parameters(path):
    with open(path, 'r') as f:
        info = f.readlines()
    del info[0]
    labels = []
    ids = []
    for i, e in enumerate(info):
        if 'eta = ' in e:
            j = e[-5:-2]
            g = e[15:18]
            eta = e[6:9]
        else:
            ids.append(e.replace('\n', ''))
            label = np.array([float(eta), float(g), float(j)], dtype=np.float32)
            labels.append(label)

    return ids, labels

ids1, labels1 = read_ids_parameters(os.path.join(top_dir, 'info.txt'))
ids2, labels2 = read_ids_parameters(os.path.join(top_dir, 'info1.txt'))
ids3, labels3 = read_ids_parameters(os.path.join(top_dir, 'info2.txt'))

labels = np.array(labels1 + labels2 + labels3)
ids = np.array(ids1 + ids2 + ids3)

## Separate out test and validation simulations
unq, indices = np.unique(labels, return_index=True, axis=0)
test_labels1 = labels[indices]
test_ids1 = ids[indices]
train_labels = np.delete(labels, indices, axis=0)
train_ids = np.delete(ids, indices, axis=0)

unq, indices = np.unique(train_labels, return_index=True, axis=0)
test_labels2 = train_labels[indices]
test_ids2 = train_ids[indices]
train_labels = np.delete(train_labels, indices, axis=0)
train_ids = np.delete(train_ids, indices, axis=0)

unq, indices = np.unique(train_labels, return_index=True, axis=0)
validation_labels = train_labels[indices]
validation_ids = train_ids[indices]
train_labels = np.delete(train_labels, indices, axis=0)
train_ids = np.delete(train_ids, indices, axis=0)
#
test_labels = np.concatenate((test_labels1, test_labels2), axis=0)
test_ids = np.concatenate((test_ids1, test_ids2), axis=0)


slice_size = 300
stride = 150
start_index = 150

full_train_im_arr = []
full_train_label_arr = []

for i, lab in enumerate(train_labels):
    images = h5_to_slices(os.path.join(top_dir, train_ids[i], 'LFP_firing_rate.h5'), slice_size, stride, start_index)
    labels = np.repeat(np.expand_dims(lab, 0), len(images), axis=0)
    full_train_im_arr.append(images[:])
    full_train_label_arr.append(labels[:])

full_train_im_arr = np.array(full_train_im_arr, dtype=np.float64).reshape([-1, 6, slice_size])
full_train_label_arr = np.array(full_train_label_arr, dtype=np.float32).reshape([-1, 3])

rand_perm_train = np.random.permutation(len(full_train_im_arr))
full_train_im_arr = full_train_im_arr[rand_perm_train]
full_train_label_arr = full_train_label_arr[rand_perm_train]

full_test_im_arr = []
full_test_label_arr = []

for i, lab in enumerate(test_labels):
    images = h5_to_slices(os.path.join(top_dir, test_ids[i], 'LFP_firing_rate.h5'), slice_size, stride, start_index)
    labels = np.repeat(np.expand_dims(lab, 0), len(images), axis=0)
    full_test_im_arr.append(images)
    full_test_label_arr.append(labels)

full_test_im_arr = np.array(full_test_im_arr, dtype=np.float64).reshape([-1, 6, slice_size])
full_test_label_arr = np.array(full_test_label_arr, dtype=np.float32).reshape([-1, 3])

rand_perm_test = np.random.permutation(len(full_test_im_arr))
full_test_im_arr = full_test_im_arr[rand_perm_test]
full_test_label_arr = full_test_label_arr[rand_perm_test]

full_validation_im_arr = []
full_validation_label_arr = []

for i, lab in enumerate(validation_labels):
    images = h5_to_slices(os.path.join(top_dir, validation_ids[i], 'LFP_firing_rate.h5'), slice_size, stride, start_index)
    labels = np.repeat(np.expand_dims(lab, 0), len(images), axis=0)
    full_validation_im_arr.append(images)
    full_validation_label_arr.append(labels)

full_validation_im_arr = np.array(full_validation_im_arr, dtype=np.float64).reshape([-1, 6, slice_size])
full_validation_label_arr = np.array(full_validation_label_arr, dtype=np.float32).reshape([-1, 3])

rand_perm_validation = np.random.permutation(len(full_validation_im_arr))
full_validation_im_arr = full_validation_im_arr[rand_perm_validation]
full_validation_label_arr = full_validation_label_arr[rand_perm_validation]

test_params, test_counts = np.unique(full_test_label_arr, return_counts=True, axis=0)
train_params, train_counts = np.unique(full_train_label_arr, return_counts=True, axis=0)
validation_params, validation_counts = np.unique(full_validation_label_arr, return_counts=True, axis=0)


full_train_im_arr -= np.mean(full_train_im_arr, axis=2)[:,:,None]
full_test_im_arr -= np.mean(full_test_im_arr, axis=2)[:,:,None]
full_validation_im_arr -= np.mean(full_validation_im_arr, axis=2)[:,:,None]

full_train_im_arr = full_train_im_arr.astype(np.float32)
full_test_im_arr = full_test_im_arr.astype(np.float32)
full_validation_im_arr = full_validation_im_arr.astype(np.float32)

create_tfrecords(os.path.join(output_path,'training_data_ngj_augmented.tfrecord'), full_train_im_arr, full_train_label_arr)
create_tfrecords(os.path.join(output_path, 'test_data_ngj_augmented.tfrecord'), full_test_im_arr, full_test_label_arr)
create_tfrecords(os.path.join(output_path, 'validation_data_ngj_augmented.tfrecord'), full_validation_im_arr, full_validation_label_arr)
