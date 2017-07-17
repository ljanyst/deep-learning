#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   06.07.2017
#-------------------------------------------------------------------------------

import tensorflow.contrib.seq2seq as seq2seq
import tensorflow as tf

from tensorflow.python.layers.core import Dense

#-------------------------------------------------------------------------------
class SortNet:
    #---------------------------------------------------------------------------
    def __init__(self, lstm_size, lstm_layers,
                 source_vocab_size, enc_embedding_size,
                 tgt_word_to_int, dec_embedding_size,
                 tgt_max_length):

        #-----------------------------------------------------------------------
        # Placeholders
        #-----------------------------------------------------------------------
        self.inputs     = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.targets    = tf.placeholder(tf.int32, [None, None], name='targets')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.tgt_seq_length = tf.placeholder(tf.int32, [None],
                                             name='tgt_seq_length')
        self.src_seq_length = tf.placeholder(tf.int32, [None],
                                             name='src_seq_length')

        #-----------------------------------------------------------------------
        # Encoder
        #-----------------------------------------------------------------------
        with tf.variable_scope('encoder'):
            with tf.variable_scope('embedding'):
                enc_embed = tf.contrib.layers.embed_sequence(self.inputs,
                              source_vocab_size, enc_embedding_size)
            with tf.variable_scope('rnn'):
                enc_cell = tf.contrib.rnn.MultiRNNCell(
                         [tf.contrib.rnn.BasicLSTMCell(lstm_size) \
                          for _ in range(lstm_layers)])

            self.initial_state = enc_cell.zero_state(self.batch_size,
                                                     tf.float32)

            _, self.enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed,
                                  sequence_length=self.src_seq_length,
                                  initial_state=self.initial_state)

        #-----------------------------------------------------------------------
        # Decoder
        #-----------------------------------------------------------------------
        target_vocab_size = len(tgt_word_to_int)
        with tf.variable_scope('decoder'):

            #-------------------------------------------------------------------
            # Embedding
            #-------------------------------------------------------------------
            with tf.variable_scope('embedding'):
                self.dec_embed = tf.Variable(
                                   tf.random_uniform([target_vocab_size,
                                                      dec_embedding_size]))

            #-------------------------------------------------------------------
            # Final classifier
            #-------------------------------------------------------------------
            with tf.variable_scope('classifier') as classifier_scope:
                self.output_layer = Dense(target_vocab_size,
                                      kernel_initializer = \
                                        tf.truncated_normal_initializer(
                                          mean = 0.0, stddev=0.1))

            #-------------------------------------------------------------------
            # RNN
            #-------------------------------------------------------------------
            with tf.variable_scope('rnn'):
                self.dec_cell = tf.contrib.rnn.MultiRNNCell(
                                  [tf.contrib.rnn.BasicLSTMCell(lstm_size) \
                                   for _ in range(lstm_layers)])

            #-------------------------------------------------------------------
            # Inference decoder
            #-------------------------------------------------------------------
            with tf.variable_scope('decoder'):
                start_tokens = tf.tile([tgt_word_to_int['<s>']],
                                       [self.batch_size])

                helper = seq2seq.GreedyEmbeddingHelper(self.dec_embed,
                                                       start_tokens,
                                                       tgt_word_to_int['</s>'])

                decoder = seq2seq.BasicDecoder(self.dec_cell, helper,
                                               self.enc_state,
                                               self.output_layer)
                outputs, _, _ = seq2seq.dynamic_decode(decoder,
                                                       impute_finished=\
                                                         True,
                                                       maximum_iterations=\
                                                         tgt_max_length)

        self.outputs = tf.identity(outputs.sample_id, 'predictions')

    #---------------------------------------------------------------------------
    def get_optimizer(self, tgt_word_to_int, tgt_max_length,
                      learning_rate=0.001):
        #-----------------------------------------------------------------------
        # Training decoder
        #-----------------------------------------------------------------------
        with tf.variable_scope('decoder', reuse=True):

            #-------------------------------------------------------------------
            # Append seqience start tags
            #-------------------------------------------------------------------
            with tf.variable_scope('preprocess'):
                fill = tf.fill([self.batch_size, 1], tgt_word_to_int['<s>'])
                dec_input = tf.concat([fill, self.targets[:, :-1]], 1)

            #-------------------------------------------------------------------
            # Embedding lookup
            #-------------------------------------------------------------------
            with tf.variable_scope('embedding', reuse=True):
                dec_vec = tf.nn.embedding_lookup(self.dec_embed, dec_input)

            #-------------------------------------------------------------------
            # Training decoder
            #-------------------------------------------------------------------
            with tf.variable_scope('decoder', reuse=True):
                helper  = seq2seq.TrainingHelper(inputs=dec_vec,
                                                 sequence_length=\
                                                   self.tgt_seq_length)
                decoder = seq2seq.BasicDecoder(self.dec_cell, helper,
                                               self.enc_state,
                                               self.output_layer)
                outputs, _, _ = seq2seq.dynamic_decode(decoder,
                                                       impute_finished=True,
                                                       maximum_iterations=\
                                                         tgt_max_length)
                logits = tf.identity(outputs.rnn_output, 'logits')

        #-----------------------------------------------------------------------
        # Define loss
        #-----------------------------------------------------------------------
        with tf.variable_scope('loss'):
            masks = tf.sequence_mask(self.tgt_seq_length, tgt_max_length,
                                     dtype=tf.float32, name='masks')
            loss  = tf.contrib.seq2seq.sequence_loss(logits, self.targets,
                                                     masks)

        #-----------------------------------------------------------------------
        # Define the loss function and the optimizer
        #-----------------------------------------------------------------------
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(loss)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) \
                                for grad, var in gradients if grad is not None]
            optimizer_op = optimizer.apply_gradients(capped_gradients)

        return optimizer_op, loss
