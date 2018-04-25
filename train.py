from convnet import inference_deep, accuracy, mse_loss, inference_shallow, inference, cross_entropy_loss, softmax_accuracy
from input_functions import load_datasets
import tensorflow as tf
import argparse
import os
import shutil
import time
import numpy as np


# Take arguments from command line
parser = argparse.ArgumentParser(description='Choose models and parameters')
parser.add_argument('n_epochs', help='number of epochs to train')
parser.add_argument('-batch', default=100, help='specifies batch size')
parser.add_argument('-clear', action='store_true', default=False)
parser.add_argument('-first', action='store_true', default=False)
parser.add_argument('-second', action='store_true', default=False)
parser.add_argument('-eta', action='store_true', default=False)
parser.add_argument('-g', action='store_true', default=False)
parser.add_argument('-j', action='store_true', default=False)
parser.add_argument('-lr', default=0.0001, help='learning rate for adam optimizer')
parser.add_argument('-reg', default=0.0, help='regularization scale')
parser.add_argument('-summary_step', default=1, help='specifies how often to log summaries')
parser.add_argument('-length',  help='specifies which signal length to use: 100, 300 or 900ms')
parser.add_argument('-do', default=1.0, help='dropout')
parser.add_argument('-acc', action='store_true', default=False)
parser.add_argument('-softmax', action='store_true', default=False)


args = parser.parse_args()

if int(args.g) + int(args.j) + int(args.eta) != 1:
    raise ValueError('Specify one parameter to train')

label_index = 0 if args.eta else 1 if args.g else 2
parameter_names = ['eta', 'g', 'J']


# Set up neccessary paths
test_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/')
training_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/')
validation_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/')

if args.softmax:
    save_top = os.path.join(
        'C:/Users/JanEirik/Documents/TensorFlow/ConvNet/save/final/classification/')
elif not args.softmax:
    save_top = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/ConvNet/save/final/regression/')

if int(args.length) == 100:
    savepath = os.path.join(save_top, '100ms/')
    test_path = os.path.join(test_path, '100ms/test_data_ngj.tfrecord')
    training_path = os.path.join(training_path, '100ms/training_data_ngj.tfrecord')
    validation_path = os.path.join(validation_path, '100ms/validation_data_ngj.tfrecord')
elif int(args.length) == 300:
    savepath = os.path.join(save_top, '300ms/')
    test_path = os.path.join(test_path, '300ms/test_data_ngj.tfrecord')
    training_path = os.path.join(training_path, '300ms/training_data_ngj.tfrecord')
    validation_path = os.path.join(validation_path, '300ms/validation_data_ngj.tfrecord')
elif int(args.length) == 900:
    savepath = os.path.join(save_top, '900ms/')
    test_path = os.path.join(test_path, '900ms/test_data_ngj.tfrecord')
    training_path = os.path.join(training_path, '900ms/training_data_ngj.tfrecord')
    validation_path = os.path.join(validation_path, '900ms/validation_data_ngj.tfrecord')
else:
    raise ValueError('Signal length must be specified to be either 100, 300 or 900')

if args.eta:
    savepath = os.path.join(savepath, 'eta/')
if args.g:
    savepath = os.path.join(savepath, 'g/')
if args.j:
    savepath = os.path.join(savepath, 'j/')


print('Training parameter %s for %d epochs, signal length %d ms' %
      (parameter_names[label_index], int(args.n_epochs), int(args.length)))

firstpath = os.path.join(savepath, 'first/')
workpath = os.path.join(savepath, 'workspace/')
workfile = os.path.join(workpath, 'model.ckpt')
firstfile = os.path.join(firstpath, 'model.ckpt')


# Some helper functions to manage files
def save_best(writer, saver, sess):
    print('New best, saving and copying files')
    writer.close()
    clear_tf_files(firstpath)
    saver.save(sess, firstfile)
    copy_tf_files(workpath, firstpath, include_models=False)
    return tf.summary.FileWriter(logdir=workpath, graph=tf.get_default_graph())


def tf_file_list(path, include_events=True, include_models=True):
    fs = os.listdir(path)
    fs = [f for f in fs if os.path.isfile(os.path.join(path, f))]
    eventfs = [f for f in fs if 'events.out.tfevents' in f]
    modelfs = [f for f in fs if 'model.ckpt' in f or 'checkpoint' in f]
    fs = eventfs*include_events + modelfs*include_models
    return fs


def clear_tf_files(path):
    dstfiles = tf_file_list(path)
    for f in dstfiles:
        os.remove(os.path.join(path, f))


def copy_tf_files(src, dst, include_models=True):
    srcfiles = tf_file_list(src, include_models=include_models)
    for f in srcfiles:
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))


def clear_copy_tf_files(src, dst, include_models=True):
    clear_tf_files(dst)
    copy_tf_files(src, dst, include_models=include_models)


if len(os.listdir(workpath)) > 0:
    clear_tf_files(workpath)

abort = False
if not (args.first or args.second or args.clear):
    save_files = os.listdir(firstpath)
    save_files = [f for f in save_files if 'model.ckpt' in f]
    if len(save_files) >= 1:
        inp = input('Save files exist, start from scratch? (y/n) ')
        if inp.lower() == 'no' or inp.lower() == 'n':
            abort = True
        elif inp.lower() == 'y' or inp.lower() == 'yes':
            print('Clearing save files')
            clear_tf_files(firstpath)


def train():
    """
    """
    # SET UP GRAPH
    # Load datasets
    saved = False
    with tf.variable_scope('data_loading'):
        x, y, train_init, test_init, val_init = load_datasets(test_path, training_path, validation_path, int(
            args.length), batch_size=int(args.batch), one_hot=args.softmax)
        if args.softmax:
            y = y[:, label_index, :]
        else:
            y = tf.expand_dims(y[:, label_index], [1])

        x = tf.transpose(x, perm=[0, 3, 2, 1])

    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    output = inference(x, phase_train, keep_prob, softmax=args.softmax)
    if args.softmax:
        loss = cross_entropy_loss(output, y)
        accuracies = softmax_accuracy(output, y)
    else:
        loss = mse_loss(output, y)
        accuracies = accuracy(output, y, j=args.j)

    with tf.variable_scope('regularization'):
        reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_sum = tf.add_n(reg_vars)
        reg_summaries = [tf.summary.scalar(reg.name, reg) for reg in reg_vars]

    # Set up optimizer
    with tf.variable_scope('optimizer'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(name='adam_optimizer', learning_rate=float(args.lr))
        optimize_step = optimizer.minimize(loss + float(args.reg)*reg_sum, global_step=global_step)

    # Declare and collect various variables, operations and summaries
    first_loss = tf.Variable(10, trainable=False, name='best_achieved_loss', dtype=tf.float32)
    test_summaries = tf.summary.merge(tf.get_collection('test_summaries'))
    training_summaries = tf.summary.merge(tf.get_collection('training_summaries'))
    validation_summaries = tf.summary.merge(tf.get_collection('validation_summaries'))
    regularization_summaries = tf.summary.merge(reg_summaries)


    running_metric_vals = tf.get_collection('running_means', scope='running_metrics')
    running_metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='running_metrics')
    running_metric_ops = tf.get_collection('running_metric_ops', scope='running_metrics')

    # Set up saver and filewriter
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logdir=workpath, graph=tf.get_default_graph())
    # Run training
    with tf.Session() as sess:
        # Initialize variables
        if not args.first:
            print('INITIALIZING')
            sess.run(tf.global_variables_initializer())
        elif args.first:
            print('LOADING FIRST NETWORK')
            copy_tf_files(firstpath, workpath)
            saver.restore(sess, workfile)

        # Train n epochs
        epoch = 0
        for i in range(int(args.n_epochs)):
            # Train one epoch
            if args.acc is not True:
                sess.run(train_init)
                print('\nTraining one epoch')
                while True:
                    try:
                        sess.run([optimize_step], feed_dict={
                                 phase_train: True, keep_prob: float(args.do)})
                    except tf.errors.OutOfRangeError:
                        break
                epoch += 1
            # Log and print accuracies and losses
            if epoch % int(args.summary_step) == 0:
                # Run through training data and update metrics
                sess.run(train_init)
                sess.run(tf.variables_initializer(running_metric_vars))
                step = tf.train.global_step(sess, global_step)
                while True:
                    try:
                        sess.run(running_metric_ops, feed_dict={phase_train: False, keep_prob: 1.0})
                    except tf.errors.OutOfRangeError:
                        break
                a = sess.run(running_metric_vals + [training_summaries])

                if args.softmax:
                    print('{:<12s}{:<12s}{:<12s}'.format('EPOCH %d' % epoch, 'loss', 'acc'))
                    print('{:<12s}{:<12.6f}{:<12.4f}'.format('Training: ', a[0], a[1]))
                else:
                    print('{:<12s}{:<12s}{:<12s}{:<12s}{:<12s}'.format('EPOCH %d' %
                                                                       epoch, 'loss', 'acc2', 'acc1', 'acc05'))
                    print('{:<12s}{:<12.6f}{:<12.4f}{:<12.4f}{:<12.4f}'.format(
                        'Training: ', a[0], a[1], a[2], a[3]))  #

                writer.add_summary(a[-1], global_step=step)
                # Run through test data
                sess.run(test_init)
                sess.run(tf.variables_initializer(running_metric_vars))
                while True:
                    try:
                        sess.run(running_metric_ops, feed_dict={phase_train: False, keep_prob: 1.0})
                    except tf.errors.OutOfRangeError:
                        break
                b = sess.run(running_metric_vals + [test_summaries])

                if args.softmax:
                    print('{:<12s}{:<12.6f}{:<12.4f}'.format('Test: ', b[0], b[1]))
                else:
                    print('{:<12s}{:<12.6f}{:<12.4f}{:<12.4f}{:<12.4f}'.format(
                        'Test: ', b[0], b[1], b[2], b[3]))
                writer.add_summary(b[-1], global_step=step)
                # Run through validation data
                sess.run(val_init)
                sess.run(tf.variables_initializer(running_metric_vars))
                while True:
                    try:
                        sess.run(running_metric_ops, feed_dict={phase_train: False, keep_prob: 1.0})
                    except tf.errors.OutOfRangeError:
                        break
                c = sess.run(running_metric_vals + [validation_summaries] + [regularization_summaries])
                if args.softmax:
                    print('{:<12s}{:<12.6f}{:<12.4f}'.format('Validation: ', c[0], c[1]))
                else:
                    print('{:<12s}{:<12.6f}{:<12.4f}{:<12.4f}{:<12.4f}'.format(
                        'Validation: ', c[0], c[1], c[2], c[3]))
                writer.add_summary(c[-1], global_step=step)
                writer.add_summary(c[-2], global_step=step)

                fl = sess.run(first_loss)
                upper_limit = 1.0 if args.softmax else 0.03
                if b[0] < fl and b[0] < upper_limit:
                    sess.run(tf.assign(first_loss, b[0]))
                    writer = save_best(writer, saver, sess)
                    saved = True
        if len(tf_file_list(firstpath)) == 0:
            saved = True
            save_best(writer, saver, sess)
        # if saved is False:
        #     inp = input('Nothing has been saved. Save current? (y/n) ')
        #     if inp.lower() == 'y' or inp.lower() == 'yes':
        #         save_best(writer, saver, sess)
        writer.close()
        clear_tf_files(workpath)


if __name__ == '__main__':
    if not abort:
        train()
