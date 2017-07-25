#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   25.07.2017
#-------------------------------------------------------------------------------

import tensorflow as tf

#-------------------------------------------------------------------------------
def LeakyReLU(x, alpha, name=None):
    return tf.maximum(alpha*x, x, name=name)

#-------------------------------------------------------------------------------
class MNISTGAN:
    #---------------------------------------------------------------------------
    def __init__(self, rnd_vec_dim, hidden_units, output_dim, alpha):
        #-----------------------------------------------------------------------
        # Inputs
        #-----------------------------------------------------------------------
        self.inputs_rnd  = tf.placeholder(tf.float32, (None, rnd_vec_dim),
                                          name='inputs_rnd')

        #-----------------------------------------------------------------------
        # The generator
        #-----------------------------------------------------------------------
        self.alpha = alpha
        with tf.variable_scope('generator'):
            h1 = tf.layers.dense(self.inputs_rnd, hidden_units, activation=None)
            h1 = LeakyReLU(h1, self.alpha)

            self.gen_logits = tf.layers.dense(h1, output_dim, activation=None)
            self.gen_out    = tf.tanh(self.gen_logits)

    #---------------------------------------------------------------------------
    def __discriminator(self, x, scope, reuse, hidden_units):
        with tf.variable_scope(scope, reuse=reuse):
            h1 = tf.layers.dense(x, hidden_units, activation=None)
            h1 = LeakyReLU(h1, self.alpha)

            logits = tf.layers.dense(h1, 1, activation=None)
            out    = tf.sigmoid(logits)

        return out, logits

    #---------------------------------------------------------------------------
    def build_discriminator(self, image_size, hidden_units):
        self.inputs_real = tf.placeholder(tf.float32, [None, image_size],
                                          name='inputs_real')

        #-----------------------------------------------------------------------
        # Process input so that it matches what the generator produces
        #-----------------------------------------------------------------------
        with tf.variable_scope('process_real'):
            processed = 2*self.inputs_real-1

        #-----------------------------------------------------------------------
        # Real discriminator
        #-----------------------------------------------------------------------
        ret = self.__discriminator(processed, 'discriminator', False,
                                   hidden_units)
        self.dsc_real_out    = ret[0]
        self.dsc_real_logits = ret[1]

        #-----------------------------------------------------------------------
        # Fake discriminator
        #-----------------------------------------------------------------------
        ret = self.__discriminator(self.gen_out, 'discriminator', True,
                                   hidden_units)
        self.dsc_fake_out    = ret[0]
        self.dsc_fake_logits = ret[1]

    #---------------------------------------------------------------------------
    def get_optimizers(self, learning_rate=0.002, smooth=0.1):
        #-----------------------------------------------------------------------
        # Define loss functions
        #-----------------------------------------------------------------------
        with tf.variable_scope('loses'):
            dsc_real_loss = tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.dsc_real_logits,
                labels=tf.ones_like(self.dsc_real_logits) * (1 - smooth)))

            dsc_fake_loss = tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.dsc_fake_logits,
                labels=tf.zeros_like(self.dsc_fake_logits)))

            dsc_loss = (dsc_real_loss + dsc_fake_loss)/2

            gen_loss = tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.dsc_fake_logits,
                labels=tf.ones_like(self.dsc_fake_logits)))

        #-----------------------------------------------------------------------
        # Optimizers
        #-----------------------------------------------------------------------
        trainable_vars = tf.trainable_variables()
        gen_vars = [var for var in trainable_vars \
                      if var.name.startswith('generator')]
        dsc_vars = [var for var in trainable_vars \
                      if var.name.startswith('discriminator')]

        with tf.variable_scope('optimizers'):
            with tf.variable_scope('deiscriminator_optimizer'):
                dsc_train_opt = tf.train.AdamOptimizer(learning_rate) \
                  .minimize(dsc_loss, var_list=dsc_vars)
            with tf.variable_scope('generator_optimizer'):
                gen_train_opt = tf.train.AdamOptimizer(learning_rate) \
                  .minimize(gen_loss, var_list=gen_vars)

        return dsc_train_opt, gen_train_opt, dsc_loss, gen_loss
