import numpy as np
import tensorflow as tf
import h5py
import os


def h5_to_slices(filepath, slice_size, stride, start_index):
    full_sig_arr = None
    with h5py.File(filepath) as hf:
        data = hf['new_data'][:,start_index:]
        file_sig_arr = np.empty(np.shape(data))
        channels = len(data[:,0])
        file_sig_arr[:] = data[:]
    n_slices = int((len(file_sig_arr[0,:])-slice_size)/stride +1)
    if full_sig_arr is None:
        full_sig_arr = np.empty([n_slices, channels, slice_size], dtype=np.float64)
    for k in range(n_slices):
        t = (k*stride, k*stride + slice_size)
        slic = file_sig_arr[:,t[0]:t[1]]
        if len(slic[0,:]) != slice_size:
            raise AttributeError('Made wrong sized slice')
        full_sig_arr[k] = slic
    return full_sig_arr

def file_list_to_slices(file_list, slice_size, stride, start_index):
    n_files = len(file_list)
    full_sig_arr = None
    for i, f in enumerate(file_list):
        file_sig_arr = h5_to_slices(f, slice_size, stride, start_index)
        n_slices = len(file_sig_arr)
        if full_sig_arr is None:
            full_sig_arr = np.empty([n_slices*n_files]+list(np.shape(file_sig_arr[0])), dtype=np.float64)
        full_sig_arr[i*n_slices:(i+1)*n_slices] = file_sig_arr
    return full_sig_arr

def create_tfrecords(file_path, images, labels):
    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    if len(images) != len(labels):
        raise AttributeError('Number of labels must be equal number of images')
    with tf.python_io.TFRecordWriter(file_path) as writer:
        for i in range(len(images)):
            feature = {'label': _bytes_feature(labels[i].tostring()),
                       'image': _bytes_feature(images[i].tostring())
                       }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
