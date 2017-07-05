#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   29.06.2017
#-------------------------------------------------------------------------------

import tensorflow as tf

#-------------------------------------------------------------------------------
class WordRNN:
    #---------------------------------------------------------------------------
    def __init__(self, n_words, embedding_size, lstm_size, lstm_layers):
        #-----------------------------------------------------------------------
        # Placeholders for input and target
        #-----------------------------------------------------------------------
        self.inputs    = tf.placeholder(tf.int32, [None, None], name="inputs")
        self.targets   = tf.placeholder(tf.int32, [None, None], name="targets")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.n_seqs    = tf.placeholder(tf.int32, (), name='n_seqs')

        #-----------------------------------------------------------------------
        # Word embedding
        #-----------------------------------------------------------------------
        with tf.variable_scope('word2vec'):
            embedding = tf.Variable(
                tf.random_uniform((n_words, embedding_size), -1, 1),
                name='embedding')

            vec = tf.nn.embedding_lookup(embedding, self.inputs)

        #-----------------------------------------------------------------------
        # The RNN
        #-----------------------------------------------------------------------
        def lstm_cell():
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            drop = tf.contrib.rnn.DropoutWrapper(
                     lstm, output_keep_prob = self.keep_prob)
            return drop

        with tf.variable_scope('rnn'):
            cell = tf.contrib.rnn.MultiRNNCell(
                     [lstm_cell() for _ in range(lstm_layers)])

            self.initial_state = cell.zero_state(self.n_seqs, tf.float32)

            rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell, vec,
                                              initial_state=self.initial_state)

        #-----------------------------------------------------------------------
        # Build the fully connected layer
        #-----------------------------------------------------------------------
        with tf.variable_scope('fully-connected'):
            self.logits = tf.contrib.layers.fully_connected(rnn_outputs,
                                                            n_words,
                                                            activation_fn=tf.identity)

        with tf.variable_scope('softmax'):
            self.output = tf.nn.softmax(self.logits, name='output')

    #---------------------------------------------------------------------------
    def get_optimizer(self, learning_rate = 0.001):
        with tf.name_scope('loss'):
            input_shape = tf.shape(self.inputs)
            ones        = tf.ones([input_shape[0], input_shape[1]])
            loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.targets,
                                                    ones)

        #-----------------------------------------------------------------------
        # Build the optimizer
        #-----------------------------------------------------------------------
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients = optimizer.compute_gradients(loss)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) \
                                for grad, var in gradients if grad is not None]
            optimizer_op = optimizer.apply_gradients(capped_gradients)

        return optimizer_op, loss
