import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pandas import DataFrame
import time

import bitcoin_data_preprocessing as reader

tf.set_random_seed(777)

flags = tf.flags
flags.DEFINE_string("save_path", "ckpt", "checkpoint_dir")
flags.DEFINE_bool("train", True, "should we train or test")
FLAGS = flags.FLAGS

class ModelConfig(object):

    def __init__(self):
        self.input_data_column_cnt = 20
        self.output_data_column_cnt = 1

        self.seq_length = 24
        self.rnn_cell_hidden_dim = 100

        # self.pattern_size = 2
        self.forget_bias = 1.0

        self.num_stacked_layers = 2
        self.keep_prob = 1.0

        self.batch_size = 514
        self.epoch_num = 2000

        self.learning_rate = 0.01

        self.max_grad_norm = 1
        self.init_scale = 0.1


class OutputConfig(object):

    def __init__(self):
        self.best_iteration = 0
        self.tr_loss = 0.0
        self.processing_time = ''


class BitLstmModel(object):
    def __init__(self, is_training, config):
        self.is_training = is_training
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length

        self.X = tf.placeholder(tf.float32, [None, self.seq_length, config.input_data_column_cnt])

        self.targets = tf.placeholder(tf.float32, [None, self.seq_length, 1])

        stackedRNNs = [self.lstm_cell(config) for _ in range(config.num_stacked_layers)]
        multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs,
                                                  state_is_tuple=True) if config.num_stacked_layers > 1 else self.lstm_cell(
            config)

        hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)

        self.hypothesis = tf.reshape(hypothesis, [self.batch_size * self.seq_length, config.rnn_cell_hidden_dim])

        softmax_w = tf.get_variable("softmax_w", [config.rnn_cell_hidden_dim, config.output_data_column_cnt],
                                    dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.output_data_column_cnt], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(self.hypothesis, softmax_w, softmax_b)
        self.logits = tf.reshape(logits, [self.batch_size, self.seq_length])


        tgg = tf.reshape(self.targets, [self.batch_size, self.seq_length])

        loss = tf.reduce_sum(tf.square(self.logits - tgg))

        self.cost = loss
        self.final_state = _states
        self.one_logit = self.logits[:, -1]
        self.one_tgg = tgg[:, -1]

        if not is_training:
            return

        optimizer = tf.train.AdamOptimizer(config.learning_rate)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())



    def lstm_cell(self, config):

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=config.rnn_cell_hidden_dim,
                                            forget_bias=config.forget_bias, state_is_tuple=True,
                                            activation=tf.nn.softsign)

        if config.keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
        return cell


def run_epoch(session, model, data_x, data_y, model_config, eval_op=None, verbose=False):


    costs = 0.0
    iters = 0
    diff = 0.0

    for step, (x, y) in enumerate(
            reader.batch_iterator(data_x, data_y, model_config.batch_size, model_config.seq_length)):
        if eval_op is not None:
            _, _cost, one_logit, one_tgg = session.run([model.train_op, model.cost, model.one_logit, model.one_tgg],
                                                       feed_dict={model.X: x, model.targets: y})
        else:
            _cost, one_logit, one_tgg = session.run([model.cost, model.one_logit, model.one_tgg],
                                                    feed_dict={model.X: x, model.targets: y})

        costs += _cost
        iters += 1
        diff += np.abs(np.sum(one_logit - one_tgg))

    return diff / iters


def main():
    today_ = datetime.date.today()
    f_path = FLAGS.save_path
    print(f_path)

    if FLAGS.train:
        try:
            if not (os.path.isdir(f_path)):
                os.makedirs(os.path.join(f_path))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise

        info_save = os.path.join(f_path, 'model_info.txt')
        print(info_save)
        f_info = open(info_save, 'w')


        model_config = ModelConfig()


        raw_data = reader.data_reader()
        dataX, dataY, data_min, data_max = reader.data_preprocessing(raw_data, model_config.seq_length)

        train_x, train_y, valid_x, valid_y = reader.data_split(dataX, dataY)

        with tf.Graph().as_default(), tf.Session() as sess:
            initializer = tf.random_uniform_initializer(-model_config.init_scale, model_config.init_scale)

            with tf.variable_scope("model", reuse=None):
                m_tr = BitLstmModel(is_training=True, config=model_config)
            with tf.variable_scope("model", reuse=True):
                m_va = BitLstmModel(is_training=False, config=model_config)

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            sum_diff = 100.0

            for epoch in range(model_config.epoch_num):
                output_config = OutputConfig()

                tr_diff = run_epoch(sess, m_tr, train_x, train_y, model_config, m_tr.train_op, verbose=True)
                va_diff = run_epoch(sess, m_va, valid_x, valid_y, model_config)


                if ((epoch + 1) % 100 == 0) or (epoch == model_config.epoch_num - 1):
                    print(
                        "epoch: {}, train_error(A): {}, valid_error(B): {}, B-A: {}".format(epoch + 1, tr_diff, va_diff,
                                                                                            va_diff - tr_diff))

                if sum_diff > ((tr_diff + va_diff) / 2):
                    sum_diff = ((tr_diff + va_diff) / 2)
                    saver.save(sess, os.path.join(f_path, 'model.ckpt'), global_step=epoch + 1)


if __name__ == "__main__":
    main()