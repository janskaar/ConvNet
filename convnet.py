import tensorflow as tf
from functools import reduce

def normalize_input(x):
    mean, var = tf.nn.moments(x, [2], keep_dims=True)
    normalized_input = tf.divide(x, tf.sqrt(var))
    return normalized_input

def conv_relu_bn(x, kernel, strides, phase_train, padding='VALID'):
    """
    kernel format: [height, width, in_channels, out_channels]
    strides format: [height, width], MUST BE LIST
    """
    _kernel = tf.get_variable(
        'kernel', kernel, initializer=tf.random_normal_initializer(stddev=0.001),
        regularizer=tf.nn.l2_loss)
    conv = tf.nn.conv2d(x, _kernel, [1] + strides + [1], padding=padding, name='conv2d')
    relu = tf.nn.relu(conv, name='relu')

    batch_mean, batch_var = tf.nn.moments(relu, [0, 1, 2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    ema_apply_op = tf.cond(phase_train, lambda: ema.apply(
        [batch_mean, batch_var]), lambda: tf.no_op())
    with tf.control_dependencies([ema_apply_op]):
        mean = tf.cond(phase_train, lambda: batch_mean, lambda: ema.average(batch_mean))
        var = tf.cond(phase_train, lambda: batch_var, lambda: ema.average(batch_var))

    beta = tf.Variable(tf.constant(0.0, shape=[kernel[3]]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[kernel[3]]), name='gamma', trainable=True)
    batch_norm = tf.nn.batch_normalization(
        relu, mean, var, beta, gamma, 1e-3, name='batch_normalization')
    return batch_norm


def conv_relu(x, kernel, strides, phase_train, padding='VALID'):
    """
    kernel format: [height, width, in_channels, out_channels]
    strides format: [height, width], MUST BE LIST
    """
    _kernel = tf.get_variable(
        'kernel', kernel, initializer=tf.random_normal_initializer(stddev=0.001),
        regularizer=tf.nn.l2_loss)
    conv = tf.nn.conv2d(x, _kernel, [1] + strides + [1], padding=padding, name='conv2d')
    relu = tf.nn.relu(conv, name='relu')
    return relu


def bn_fc_relu(x, N_in, N, phase_train, keep_prob=None):
    """
    N_in: number of neurons in previous layer
    N: number of neurons in this layer
    kernel format: [height, width, in_channels, out_channels]
    strides format: [height, width], MUST BE LIST
    """
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    ema_apply_op = tf.cond(phase_train, lambda: ema.apply(
        [batch_mean, batch_var]), lambda: tf.no_op())
    with tf.control_dependencies([ema_apply_op]):
        mean = tf.cond(phase_train, lambda: batch_mean, lambda: ema.average(batch_mean))
        var = tf.cond(phase_train, lambda: batch_var, lambda: ema.average(batch_var))

    beta = tf.Variable(tf.constant(0.0, shape=[N_in]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[N_in]), name='gamma', trainable=True)
    batch_norm = tf.nn.batch_normalization(
        x, mean, var, beta, gamma, 1e-3, name='batch_normalization')
    weights = tf.get_variable('weights', [N_in, N],
                              initializer=tf.random_normal_initializer(stddev=0.001),
                              regularizer=tf.nn.l2_loss)
    fc_h = tf.matmul(batch_norm, weights, name='matmul')
    relu = tf.nn.relu(fc_h, name='relu')
    if keep_prob is None:
        return relu
    else:
        dropout = tf.nn.dropout(relu, keep_prob=keep_prob)
        return dropout


def fc_relu(x, N_in, N, phase_train=None):
    weights = tf.get_variable('weights', [N_in, N],
                              initializer=tf.random_normal_initializer(stddev=0.001),
                              regularizer=tf.nn.l2_loss)
    fc_h = tf.matmul(x, weights, name='matmul')
    relu = tf.nn.relu(fc_h, name='relu')
    return relu


def max_pool(x, ksize, strides, padding='VALID', flatten=False):
    """
    Max pool layer
    ksize format: same as input, [N, H, W, C]
    stride format: same as input, [N, H, W, C]
    """
    max_pool = tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)
    if flatten:
        length = reduce(lambda y, z: y*z, max_pool.shape.as_list()[1:])
        max_pool = tf.reshape(max_pool, [-1,  length])
        return max_pool, length
    else:
        return max_pool

def output(x, N_in, N, phase_train):
    weights = tf.get_variable('weights', [N_in, N],
                              initializer=tf.random_normal_initializer(stddev=0.001))
    bias = tf.get_variable('bias', [N])
    y = tf.matmul(x, weights, name='output') + bias
    return y

def mse_loss(x, y):
    """
    y: labels
    x: predictions
    """
    with tf.variable_scope('loss'):
        loss = tf.losses.mean_squared_error(y, x)
    with tf.variable_scope('running_metrics/'):
        streaming_loss, streaming_loss_op = tf.metrics.mean(loss, name='streaming_loss', metrics_collections=[
                                                            'running_means'], updates_collections=['running_metric_ops'])
    tf.summary.scalar('training_loss', streaming_loss, collections=['training_summaries'])
    tf.summary.scalar('test_loss', streaming_loss, collections=['test_summaries'])
    tf.summary.scalar('validation_loss', streaming_loss, collections=['validation_summaries'])
    return loss


def cross_entropy_loss(output, label):
    with tf.variable_scope('loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=label, logits=output, name='softmax_cross_entropy')
    with tf.variable_scope('running_metrics/'):
        streaming_loss, streaming_loss_op = tf.metrics.mean(loss, name='streaming_loss', metrics_collections=[
                                                            'running_means'], updates_collections=['running_metric_ops'])
    tf.summary.scalar('training_loss', streaming_loss, collections=['training_summaries'])
    tf.summary.scalar('test_loss', streaming_loss, collections=['test_summaries'])
    tf.summary.scalar('validation_loss', streaming_loss, collections=['validation_summaries'])
    return loss

def accuracy(x, y, j=False):
    """
    Calculates accuracy of predictions at +- 0.1, 0.05, 0.01
    """
    with tf.variable_scope('accuracies'):
        difference = tf.subtract(y, x, name='prediction_errors')
        abs_diff = tf.abs(difference)
        errors = []
        if j:
            errors.append(tf.reduce_mean(tf.cast(tf.less_equal(abs_diff, 0.02), tf.float32)))
            errors.append(tf.reduce_mean(tf.cast(tf.less_equal(abs_diff, 0.01), tf.float32)))
            errors.append(tf.reduce_mean(tf.cast(tf.less_equal(abs_diff, 0.005), tf.float32)))
        else:
            errors.append(tf.reduce_mean(tf.cast(tf.less_equal(abs_diff, 0.2), tf.float32)))
            errors.append(tf.reduce_mean(tf.cast(tf.less_equal(abs_diff, 0.1), tf.float32)))
            errors.append(tf.reduce_mean(tf.cast(tf.less_equal(abs_diff, 0.05), tf.float32)))

    with tf.variable_scope('running_metrics/'):
        accuracy2, accuracy2_op = tf.metrics.mean(errors[0], name='accuracy_2', metrics_collections=[
                                                      'running_means'], updates_collections=['running_metric_ops'])
        accuracy1, accuracy1_op = tf.metrics.mean(errors[1], name='accuracy_1', metrics_collections=[
                                                        'running_means'], updates_collections=['running_metric_ops'])
        accuracy05, accuracy05_op = tf.metrics.mean(errors[2], name='accuracy_05', metrics_collections=[
                                                        'running_means'], updates_collections=['running_metric_ops'])

    with tf.variable_scope('accuracy_summaries'):
        tf.summary.scalar(
            'test_accuracy_0.02' if j else 'test_accuracy_0.2', accuracy2,
            collections=['test_summaries'])
        tf.summary.scalar(
            'test_accuracy_0.01' if j else 'test_accuracy_0.1', accuracy1,
                          collections=['test_summaries'])
        tf.summary.scalar(
            'test_accuracy_0.005' if j else 'test_accuracy_0.05', accuracy05,
                          collections=['test_summaries'])
        tf.summary.scalar(
            'training_accuracy_0.02' if j else 'training_accuracy_0.2', accuracy2,
                          collections=['training_summaries'])
        tf.summary.scalar(
            'training_accuracy_0.01' if j else 'training_accuracy_0.1', accuracy1,
                          collections=['training_summaries'])
        tf.summary.scalar(
            'training_accuracy_0.005' if j else 'training_accuracy_0.05', accuracy05,
                          collections=['training_summaries'])
        tf.summary.scalar(
            'validation_accuracy_0.02' if j else 'validation_accuracy_0.2', accuracy2,
                          collections=['validation_summaries'])
        tf.summary.scalar(
            'validation_accuracy_0.01' if j else 'validation_accuracy_0.1', accuracy1,
                          collections=['validation_summaries'])
        tf.summary.scalar(
            'validation_accuracy_0.005' if j else 'validation_accuracy_0.05', accuracy05,
                          collections=['validation_summaries'])
    return errors


def softmax_accuracy(x, y):
    correct_predictions = tf.equal(tf.argmax(x, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    with tf.variable_scope('running_metrics/'):
        accuracy, accuracy_op = tf.metrics.mean(accuracy, name='accuracy',
                                                metrics_collections=[
                                                            'running_means'],
                                                updates_collections=[
                                                            'running_metric_ops'])
    with tf.variable_scope('accuracy_summaries'):
        tf.summary.scalar('test_accuracy', accuracy,
                          collections=['test_summaries'])
        tf.summary.scalar('training_accuracy', accuracy,
                          collections=['training_summaries'])
        tf.summary.scalar('validation_accuracy', accuracy,
                          collections=['validation_summaries'])

    return accuracy


def inference(x, phase_train, keep_prob, softmax=False):
    with tf.variable_scope('ConvReluBN1'):
        y = conv_relu_bn(x, kernel=[1, 12, 6, 256], strides=[1, 1],
                         phase_train=phase_train, padding='SAME')
    with tf.variable_scope('ConvReluBN2'):
        y = conv_relu_bn(y, kernel=[1, 3, 256, 256], strides=[1, 1],
                         phase_train=phase_train, padding='SAME')
    with tf.variable_scope('max_pool1'):
        y = max_pool(y, [1, 1, 2, 1], [1, 1, 2, 1])
    with tf.variable_scope('ConvReluBN3'):
        y = conv_relu_bn(y, kernel=[1, 3, 256, 512], strides=[1, 1],
                         phase_train=phase_train, padding='SAME')
    with tf.variable_scope('ConvReluBN4'):
        y = conv_relu_bn(y, kernel=[1, 3, 512, 512], strides=[1, 1],
                         phase_train=phase_train, padding='SAME')
    with tf.variable_scope('max_pool2'):
        y = max_pool(y, [1, 1, 2, 1], [1, 1, 2, 1])
    with tf.variable_scope('ConvReluBN5'):
        y = conv_relu(y, kernel=[1, 3, 512, 512], strides=[1, 1],
                      phase_train=phase_train, padding='SAME')
    with tf.variable_scope('ConvReluBN6'):
        y = conv_relu(y, kernel=[1, 3, 512, 512], strides=[1, 1],
                      phase_train=phase_train, padding='SAME')
    with tf.variable_scope('max_pool3'):
        y = max_pool(y, [1, 1, 2, 1], [1, 1, 2, 1])
    with tf.variable_scope('ConvReluBN7'):
        y = conv_relu(y, kernel=[1, 3, 512, 1024], strides=[1, 1],
                      phase_train=phase_train, padding='SAME')
    with tf.variable_scope('ConvReluBN8'):
        y = conv_relu(y, kernel=[1, 3, 1024, 1024], strides=[1, 1],
                      phase_train=phase_train, padding='SAME')
    with tf.variable_scope('max_pool4'):
        y, length = max_pool(y, [1, 1, 2, 1], [1, 1, 2, 1], flatten=True)
    with tf.variable_scope('BNDenseRelu1'):
        size1 = 2048
        y = bn_fc_relu(y, length, size1, phase_train=phase_train, keep_prob=keep_prob)
    with tf.variable_scope('BNDenseRelu2'):
        size2 = 2048
        y = bn_fc_relu(y, size1, size2, phase_train=phase_train, keep_prob=None)
    with tf.variable_scope('output'):
        if softmax:
            y = output(y, size2, 8, phase_train=phase_train)
        else:
            y = output(y, size2, 1, phase_train=phase_train)
        return y
