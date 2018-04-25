from ConvNet.hdf5_to_tfrecords import h5_to_slices
import tensorflow as tf
import numpy as np
import os

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

def decode_dataset(element, length, one_hot=False, etas=None, gs=None, js=None):
    features = {'label': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string)
                }
    parsed = tf.parse_single_example(element, features=features)
    im = tf.decode_raw(parsed['image'], out_type=tf.float32)
    im_reshaped = tf.reshape(im, [6, length])
    label = tf.decode_raw(parsed['label'], out_type=tf.float32)
    if one_hot:
        label = tf.squeeze(tf.one_hot([tf.where(tf.equal(label[0], etas)), tf.where(tf.equal(label[1], gs)), tf.where(tf.equal(label[2], js))], 8))
    return im_reshaped, label

def load_datasets(test_path, train_path, validation_path, length, batch_size=100, one_hot=False):
    if one_hot:
        etas = tf.Variable([1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0], trainable=False)
        gs = tf.Variable([4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4], trainable=False)
        js = tf.Variable([0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2], trainable=False)
    else:
        etas = None
        gs = None
        js = None
    test_set = tf.data.TFRecordDataset(test_path)
    test_set = test_set.map(lambda x: decode_dataset(x, length, one_hot=one_hot, etas=etas, gs=gs, js=js))
    test_set = test_set.batch(batch_size)

    train_set = tf.data.TFRecordDataset(train_path)
    train_set = train_set.map(lambda x: decode_dataset(x, length, one_hot=one_hot, etas=etas, gs=gs, js=js))
    train_set = train_set.shuffle(100000)
    train_set = train_set.batch(batch_size)

    validation_set = tf.data.TFRecordDataset(validation_path)
    validation_set = validation_set.map(lambda x: decode_dataset(x, length, one_hot=one_hot, etas=etas, gs=gs, js=js))
    validation_set = validation_set.batch(batch_size)

    iterator = tf.contrib.data.Iterator.from_structure(train_set.output_types, train_set.output_shapes)
    x, y = iterator.get_next()
    x_reshaped = tf.reshape(x, [-1, 6, length, 1])
    if one_hot:
        y_reshaped = tf.reshape(y, [-1, 3, 8])
    else:
        y_reshaped = tf.reshape(y, [-1, 3])
    train_init_op = iterator.make_initializer(train_set)
    test_init_op = iterator.make_initializer(test_set)
    validation_init_op = iterator.make_initializer(validation_set)
    return x_reshaped, y_reshaped, train_init_op, test_init_op, validation_init_op
