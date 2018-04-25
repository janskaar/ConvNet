import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from convnet import inference
import os


save_dir = os.path.join('../parametersearch/g/0/best/')



x = np.linspace(0, 75, 300)
sins = np.array([np.sin(x) for i in range(6)])

image = tf.get_variable('image', [1, 1, 300, 6], initializer=tf.random_normal_initializer(stddev=0.5))
label = tf.placeholder(tf.float32, name='label')
phase_train = tf.placeholder(tf.bool, name='phase_train')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

output = inference(image, phase_train, keep_prob, softmax=False)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.1)


variables = tf.global_variables()
variables = [variable for variable in variables if 'image' not in variable.name]

saver = tf.train.Saver(var_list=variables)

map_index = 0
with tf.Session() as sess:
    saver.restore(sess, os.path.join(save_dir, 'model.ckpt'))

    feature_map = tf.get_default_graph().get_tensor_by_name('ConvReluBN8/conv2d:0')
    loss = tf.multiply(tf.reduce_mean(feature_map[0,0,:,map_index]), -1)
    optimize_step = optimizer.minimize(loss, var_list=[image])

    ## Get hidden Adam variables
    slots = optimizer.get_slot_names()
    slotvars = []
    for slot in slots:
        slotvars.append(optimizer.get_slot(image, slot))
    beta1, beta2 = optimizer._get_beta_accumulators()

    sess.run(tf.variables_initializer(slotvars + [image] + [beta1, beta2]))

    print(sess.run(tf.reduce_mean(feature_map[0,0,:,map_index]), feed_dict={phase_train:False}))
    image_list = []
    original_list = []
    original_list.append(sess.run(image))
    for i in range(5):
        orig = sess.run(image)
        for j in range(1000):
            sess.run([optimize_step], feed_dict={phase_train:False})
        im = sess.run(image)
        image_list.append(im)
        print(sess.run(tf.reduce_mean(feature_map[0,0,:,map_index]), feed_dict={phase_train:False}))
        sess.run(tf.variables_initializer([image]))
        image_list.append(im)
        original_list.append(orig)
    # sess.run(tf.assign(image, tf.add(image, tf.random_normal([1,1,300,6], stddev=0.2))))

fig, ax = plt.subplots(5, sharex=True)
for i in range(5):
    ax[i].plot(image_list[i][0,0,:,3])

fig1, ax1 = plt.subplots(5, sharex=True)
for i in range(5):
    ax1[i].plot(original_list[i][0,0,:,3])
plt.show()
