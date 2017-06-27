#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   26.06.2017
#-------------------------------------------------------------------------------

import tensorflow as tf

#-------------------------------------------------------------------------------
class SentimentRNN:
    #---------------------------------------------------------------------------
    def __init__(self, batch_size, lstm_size, lstm_layers, embedding,
                 vocab_size=0):
        #-----------------------------------------------------------------------
        # Placeholders for inputs and targets
        #-----------------------------------------------------------------------
        if embedding is None:
            self.embedding = tf.Variable(
                tf.random_uniform((vocab_size, 300), -1, 1),
                name='embedding')
        else:
            self.embedding = tf.placeholder(tf.float32, embedding.shape,
                                            name='embedding')
        self.input     = tf.placeholder(tf.int32, (None, None), name="input")
        self.target    = tf.placeholder(tf.int32, (None, None), name="target")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        #-----------------------------------------------------------------------
        # Build the LSTM layers and the RNN
        #-----------------------------------------------------------------------
        with tf.variable_scope('input-encode'):
            word2vec = tf.nn.embedding_lookup(self.embedding, self.input)

        def lstm_cell():
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            drop = tf.contrib.rnn.DropoutWrapper(
                     lstm, output_keep_prob = self.keep_prob)
            return drop

        with tf.variable_scope('rnn'):
            cell = tf.contrib.rnn.MultiRNNCell(
                     [lstm_cell() for _ in range(lstm_layers)])

            self.initial_state = cell.zero_state(batch_size, tf.float32)

            rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell, word2vec,
                                              initial_state=self.initial_state)

        #-----------------------------------------------------------------------
        # Build the fully connected layer
        #-----------------------------------------------------------------------
        with tf.variable_scope('fully-connected'):
            self.output = tf.contrib.layers.fully_connected(rnn_outputs[:, -1],
                            1, activation_fn=tf.sigmoid)

    #---------------------------------------------------------------------------
    def get_optimizer(self, learning_rate = 0.001, grad_clip = 5):
        #-----------------------------------------------------------------------
        # Build a loss function
        #-----------------------------------------------------------------------
        with tf.variable_scope('loss'):
            loss = tf.losses.mean_squared_error(self.target, self.output)

        #-----------------------------------------------------------------------
        # Build the optimizer
        #-----------------------------------------------------------------------
        with tf.variable_scope('optimizer'):
            tvars     = tf.trainable_variables()
            grads, _  = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                               grad_clip)
            train_op  = tf.train.AdamOptimizer(learning_rate)
            optimizer = train_op.apply_gradients(zip(grads, tvars))

        return optimizer, loss
