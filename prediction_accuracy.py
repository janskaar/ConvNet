import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from convnet import inference_deep, mse_loss
from input_functions import load_datasets
import os

## 0 = eta, 1 = g, 2 = J
PARAMETER = 2
paramname = 'eta' if PARAMETER == 0 else 'g' if PARAMETER == 1 else 'j'


test_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/delta_poisson/2/test_data_ngj.tfrecord')
training_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/delta_poisson/2/training_data_ngj.tfrecord')

savefile = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/ConvNet/save/%d/first/model.ckpt'%(PARAMETER+3))

output_folder = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/ConvNet/plotting/parameterspace_accuracy')

## Keep dtype float32 to match TF
etas = np.array([1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0], dtype=np.float32)
gs = np.array([4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4], dtype=np.float32)
js = np.array([0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2], dtype=np.float32)

train_accuracy_arr = np.zeros([len(etas), len(gs), len(js)])
train_npass_arr = np.zeros([len(etas), len(gs), len(js)], dtype=np.int64)
test_accuracy_arr = np.zeros([len(etas), len(gs), len(js)])
test_npass_arr = np.zeros([len(etas), len(gs), len(js)], dtype=np.int64)

x, y, train_init, test_init = load_datasets(test_path, training_path, batch_size=100)
phase_train = tf.placeholder(tf.bool, name='phase_train')
output = inference_deep(x, phase_train)
loss = mse_loss(output, tf.expand_dims(y[:,PARAMETER], [1]))

avg_train_loss = np.array([0], dtype=np.float32)
train_pass = 0
avg_test_loss = np.array([0], dtype=np.float32)
test_pass = 0

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, savefile)
    sess.run(train_init)
    while True:
        try:
            a = sess.run([output, y, loss], feed_dict={phase_train: False})
            error = np.absolute(np.subtract(a[0][:,0], a[1][:,PARAMETER]))
            avg_train_loss += a[2]
            train_pass += 1
            for i, param in enumerate(a[1]):
                etapos = np.where(etas == param[0])
                gpos = np.where(gs == param[1])
                jpos = np.where(js == param[2])
                train_accuracy_arr[etapos, gpos, jpos] += error[i]
                train_npass_arr[etapos, gpos, jpos] += 1
        except tf.errors.OutOfRangeError:
            break
    print(np.shape(a[0]), np.shape(a[1][:,PARAMETER]), np.shape(error))
    # print([a[0], a[1][:,PARAMETER], error])
    sess.run(test_init)
    while True:
        try:
            a = sess.run([output, y, loss], feed_dict={phase_train: False})
            error = np.absolute(np.subtract(a[0][:,0], a[1][:,PARAMETER]))
            avg_test_loss += a[2]
            test_pass += 1
            for i, param in enumerate(a[1]):
                etapos = np.where(etas == param[0])
                gpos = np.where(gs == param[1])
                jpos = np.where(js == param[2])
                test_accuracy_arr[etapos, gpos, jpos] += error[i]
                test_npass_arr[etapos, gpos, jpos] += 1
        except tf.errors.OutOfRangeError:
            break
    # print([a[0], a[1][:,PARAMETER].T, error])
    print(np.shape(a[0]), np.shape(a[1][:,PARAMETER]), np.shape(error))

avg_train_loss /= train_pass
avg_test_loss /= test_pass
n_notthirty = 0
for (eta, g, j), n in np.ndenumerate(train_npass_arr):
    if n != 30:
        n_notthirty += 1
    train_accuracy_arr[eta, g, j] /= n

n_notsix = 0
for (eta, g, j), n in np.ndenumerate(test_npass_arr):
    if n != 6:
        n_notsix += 1
    test_accuracy_arr[eta, g, j] /= n

print(n_notthirty, n_notsix)
print(np.mean(train_accuracy_arr)**2, avg_train_loss)
print(np.mean(test_accuracy_arr)**2, avg_test_loss)

np.save(os.path.join(output_folder, 'training_accuracy_'+paramname), train_accuracy_arr)
np.save(os.path.join(output_folder, 'test_accuracy_'+paramname), test_accuracy_arr)
