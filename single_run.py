from convnet import accuracy, mse_loss, softmax_accuracy, cross_entropy_loss, inference
from input_functions import load_datasets
import tensorflow as tf
import os, shutil, sys

class Run():
    def __init__(self, training_path, test_path, validation_path, lab_index, savepath, lr, reg, keep_prob,
                 save_step = 1, batch_size=100, length=300, load=False, softmax=False, decay=0.97, decay_step=400,
                 momentum=0.9):
        self.label_index = lab_index                            ## 0 for eta, 1 for g, 2 for j
        self.workpath = os.path.join(savepath, 'workspace/')
        self.bestpath = os.path.join(savepath, 'best/')
        self.savefile = os.path.join(self.bestpath, 'model.ckpt')
        self.test_path = test_path
        self.training_path = training_path
        self.validation_path = validation_path
        # self.lr = tf.Variable(lr, trainable=False, name='learning_rate')
        self.lr = lr
        self.reg = reg
        self.dropout = keep_prob
        self.save_epoch = 0
        self.atpt = 1
        self.save_step = save_step
        self.batch_size = batch_size
        self.length = length
        self.epoch = 0
        self.load = load
        self.softmax = softmax
        self.p_best_loss = 10
        self.decay_step = decay_step
        self.decay = decay
        self.momentum = momentum
        self.setup_graph()

    def setup_graph(self):
        with tf.variable_scope('data_loading'):
            x, y, self.train_init, self.test_init, self.val_init = load_datasets(
                                                                        self.test_path,
                                                                        self.training_path,
                                                                        self.validation_path,
                                                                        self.length,
                                                                        batch_size=self.batch_size,
                                                                        one_hot=self.softmax)
            if self.softmax:
                self.y = y[:, self.label_index, :]
            else:
                self.y = tf.expand_dims(y[:, self.label_index], [1])
            x = tf.transpose(x, perm=[0, 3, 2, 1])

        self.best_loss = tf.Variable(10, trainable=False, dtype=tf.float32, name='best_loss')
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.output = inference(x, self.phase_train, self.keep_prob, softmax=self.softmax)
        if self.softmax:
            self.loss = cross_entropy_loss(self.output, self.y)
            self.accuracies = softmax_accuracy(self.output, self.y)
        else:
            self.loss = mse_loss(self.output, self.y)
            self.accuracies = accuracy(self.output, self.y, j=(self.label_index == 2))

        with tf.variable_scope('regularization'):
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_sum = tf.add_n(reg_vars)
            reg_summaries = [tf.summary.scalar(reg.name, reg) for reg in reg_vars]

        with tf.variable_scope('optimizer'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.lr = tf.train.exponential_decay(self.lr, self.global_step, self.decay_step,
                                                 self.decay, name='learning_rate')
            tf.summary.scalar('learning_rate', self.lr, collections=['training_summaries'])
            # optimizer = tf.train.AdamOptimizer(name='adam_optimizer', learning_rate=self.lr, epsilon=self.epsilon,
            #                                    beta1=self.beta1, beta2=self.beta2)
            optimizer = tf.train.MomentumOptimizer(name='momentumoptimizer', learning_rate=self.lr, momentum=self.momentum,
                                                   use_nesterov=True)
            self.optimize_step = optimizer.minimize(self.loss + self.reg*reg_sum, global_step=self.global_step)

        self.test_summaries = tf.summary.merge(tf.get_collection('test_summaries'))
        sums = tf.get_collection('training_summaries')
        print(sums)
        self.training_summaries = tf.summary.merge(tf.get_collection('training_summaries'))
        self.validation_summaries = tf.summary.merge(tf.get_collection('validation_summaries'))
        self.regularization_summaries = tf.summary.merge(reg_summaries)
        self.running_metric_vals = tf.get_collection('running_means', scope='running_metrics')
        self.running_metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='running_metrics')
        self.running_metric_ops = tf.get_collection('running_metric_ops', scope='running_metrics')
        self.saver = tf.train.Saver()

    def train(self):
        with tf.Session() as sess:
            if self.load:
                self.load_best(sess)
                self.writer = tf.summary.FileWriter(logdir=self.workpath)
                self.report(sess)
            else:
                sess.run(tf.global_variables_initializer())
                self.writer = tf.summary.FileWriter(logdir=self.workpath, graph=tf.get_default_graph())
                self.report(sess)
            while True:
                sess.run([self.train_init])
                while True:
                    try:
                        sess.run(self.optimize_step, feed_dict={self.phase_train: True, self.keep_prob: self.dropout})
                    except tf.errors.OutOfRangeError:
                        break
                self.epoch += 1
                if self.epoch%self.save_step == 0:
                    self.report(sess)
                    if self.epoch - self.save_epoch > 20 and self.atpt > 1:
                        break
                    if self.epoch - self.save_epoch > 20 and self.atpt <= 1:
                        print('NEW ATTEMPT')
                        print('SAVE EPOCH', self.save_epoch)
                        self.writer.close()
                        self.atpt += 1
                        if len(os.listdir(self.bestpath)) > 0:
                            print('RELOADING BEST')
                            self.load_best(sess)
                            self.writer = tf.summary.FileWriter(logdir=self.workpath)
                        else:
                            print('RE-INITIALIZING')
                            self.atpt += 1
                            self.epoch = 0
                            self.clear_tf_files(self.workpath)
                            sess.run(tf.global_variables_initializer())
                            self.writer = tf.summary.FileWriter(logdir=self.workpath, graph=tf.get_default_graph())
            self.writer.close()

    def report(self, sess):
        print('\nSAVE EPOCH', self.save_epoch)
        print('Checking loss and accuracy')
        sess.run(self.train_init)
        sess.run(tf.variables_initializer(self.running_metric_vars))
        step = tf.train.global_step(sess, self.global_step)
        while True:
            try:
                sess.run(self.running_metric_ops, feed_dict={self.phase_train: False, self.keep_prob: 1.0})
            except tf.errors.OutOfRangeError:
                break
        training_results = sess.run(self.running_metric_vals + [self.training_summaries])
        sess.run(self.test_init)
        sess.run(tf.variables_initializer(self.running_metric_vars))
        while True:
            try:
                sess.run(self.running_metric_ops, feed_dict={self.phase_train: False, self.keep_prob: 1.0})
            except tf.errors.OutOfRangeError:
                break
        test_results = sess.run(self.running_metric_vals + [self.test_summaries])
        sess.run(self.val_init)
        sess.run(tf.variables_initializer(self.running_metric_vars))
        while True:
            try:
                sess.run(self.running_metric_ops, feed_dict={self.phase_train: False, self.keep_prob: 1.0})
            except tf.errors.OutOfRangeError:
                break
        validation_results = sess.run(self.running_metric_vals + [self.validation_summaries, self.regularization_summaries])
        self.print_results(training_results, test_results, validation_results, self.epoch)
        self.writer.add_summary(training_results[-1], global_step=step)
        self.writer.add_summary(test_results[-1], global_step=step)
        self.writer.add_summary(validation_results[-1], global_step=step)
        self.writer.add_summary(validation_results[-2], global_step=step)
        bl = sess.run(self.best_loss)
        print(bl)
        if test_results[0] < bl and test_results[0] < 1.5:
            sess.run(tf.assign(self.best_loss, test_results[0]))
            self.p_best_loss = bl
            self.save(sess)
            self.atpt = 1

    def save(self, sess):
        print('New best, saving and copying files')
        self.save_epoch = self.epoch
        self.writer.close()
        self.clear_tf_files(self.bestpath)
        self.saver.save(sess, self.savefile)
        self.copy_tf_files(self.workpath, self.bestpath, include_models=False)
        self.writer = tf.summary.FileWriter(logdir=self.workpath)

    def load_best(self, sess):
        self.epoch = self.save_epoch
        self.clear_tf_files(self.workpath)
        self.copy_tf_files(self.bestpath, self.workpath, include_models=False)
        self.saver.restore(sess, self.savefile)

    def tf_file_list(self, path, include_events=True, include_models=True):
        fs = os.listdir(path)
        fs = [f for f in fs if os.path.isfile(os.path.join(path, f))]
        eventfs = [f for f in fs if 'events.out.tfevents' in f]
        modelfs = [f for f in fs if 'model.ckpt' in f or 'checkpoint' in f]
        fs = eventfs*include_events + modelfs*include_models
        return fs

    def clear_tf_files(self, path):
        dstfiles = self.tf_file_list(path)
        for f in dstfiles:
            os.remove(os.path.join(path, f))

    def copy_tf_files(self, src, dst, include_models=True):
        srcfiles = self.tf_file_list(src, include_models=include_models)
        for f in srcfiles:
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))

    def print_results(self, training, test, val, epoch):
        if self.softmax:
            print('{:<12s}{:<12s}{:<12s}'.format(
                'EPOCH %d'%epoch, 'loss', 'acc'))
            print('{:<12s}{:<12.6f}{:<12.4f}'.format(
                'Training: ', training[0], training[1]))
            print('{:<12s}{:<12.6f}{:<12.4f}'.format(
                'Test: ', test[0], test[1]))
            print('{:<12s}{:<12.6f}{:<12.4f}'.format(
                'Validation: ', val[0], val[1]))
        else:
            print('{:<12s}{:<12s}{:<12s}{:<12s}{:<12s}'.format(
                'EPOCH %d'%epoch, 'loss', 'acc2', 'acc1', 'acc05'))
            print('{:<12s}{:<12.6f}{:<12.4f}{:<12.4f}{:<12.4f}'.format(
                'Training: ', training[0], training[1], training[2], training[3]))
            print('{:<12s}{:<12.6f}{:<12.4f}{:<12.4f}{:<12.4f}'.format(
                'Test: ', test[0], test[1], test[2], test[3]))
            print('{:<12s}{:<12.6f}{:<12.4f}{:<12.4f}{:<12.4f}'.format(
                'Validation: ', val[0], val[1], val[2], val[3]))




1
