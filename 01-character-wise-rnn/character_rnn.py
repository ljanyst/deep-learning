#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   02.06.2017
# Based on https://github.com/udacity/deep-learning/tree/master/intro-to-rnns
#-------------------------------------------------------------------------------

import tensorflow as tf

#-------------------------------------------------------------------------------
class CharacterRNN:
    #---------------------------------------------------------------------------
    def __init__(self, n_classes, n_seqs, n_steps, lstm_size, lstm_layers):
        #-----------------------------------------------------------------------
        # Placeholders for inputs and targets
        #-----------------------------------------------------------------------
        self.inputs    = tf.placeholder(tf.int32, (n_seqs, n_steps), \
                                        name="inputs")
        self.targets   = tf.placeholder(tf.int32, (n_seqs, n_steps), \
                                        name="targets")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        #-----------------------------------------------------------------------
        # Build the LSTM layers and the RNN
        #-----------------------------------------------------------------------
        with tf.name_scope('inputs-encode'):
            x_one_hot = tf.one_hot(self.inputs, n_classes)

        def lstm_cell():
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            drop = tf.contrib.rnn.DropoutWrapper(
                     lstm, output_keep_prob = self.keep_prob)
            return drop

        with tf.name_scope('rnn'):
            cell = tf.contrib.rnn.MultiRNNCell(
                     [lstm_cell() for _ in range(lstm_layers)])

            self.initial_state = cell.zero_state(n_seqs, tf.float32)

            rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell, x_one_hot,
                                              initial_state=self.initial_state)

        #-----------------------------------------------------------------------
        # Build the fully connected layer
        #-----------------------------------------------------------------------
        with tf.name_scope('rnn-reshape'):
            reshaped = tf.reshape(rnn_outputs, [-1, lstm_size])

        with tf.name_scope('fully-connected'):
            logits_w = tf.Variable(tf.truncated_normal((lstm_size, n_classes),
                                                       stddev=0.1))
            logits_b = tf.Variable(tf.zeros(n_classes))
            self.logits = tf.matmul(reshaped, logits_w) + logits_b
            tf.summary.histogram('logits_b', logits_b)
            tf.summary.histogram('logits_w', logits_w)

        with tf.name_scope('outputs'):
            self.outputs = tf.nn.softmax(self.logits, name='predictions')

        #-----------------------------------------------------------------------
        # Set up some other members
        #-----------------------------------------------------------------------
        self.n_classes = n_classes

    #---------------------------------------------------------------------------
    def get_training_tensors(self, learning_rate = 0.001, grad_clip = 5):
        #-----------------------------------------------------------------------
        # Build a loss function
        #-----------------------------------------------------------------------
        with tf.name_scope('targets-encode'):
            y_one_hot  = tf.one_hot(self.targets, self.n_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())

        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                           labels=y_reshaped)
            loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss', loss)

        #-----------------------------------------------------------------------
        # Build the optimizer
        #-----------------------------------------------------------------------
        with tf.name_scope('optimizer'):
            tvars     = tf.trainable_variables()
            grads, _  = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                               grad_clip)
            train_op  = tf.train.AdamOptimizer(learning_rate)
            optimizer = train_op.apply_gradients(zip(grads, tvars))

        return loss, optimizer
