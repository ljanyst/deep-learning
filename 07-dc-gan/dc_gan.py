#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   26.07.2017
#-------------------------------------------------------------------------------

import tensorflow as tf

#-------------------------------------------------------------------------------
def LeakyReLU(x, alpha, name=None):
    return tf.maximum(alpha*x, x, name=name)

#-------------------------------------------------------------------------------
class DCGAN:
    #---------------------------------------------------------------------------
    def __init__(self, rnd_vec_dim, alpha):
        #-----------------------------------------------------------------------
        # Inputs
        #-----------------------------------------------------------------------
        self.inputs_rnd  = tf.placeholder(tf.float32, (None, rnd_vec_dim),
                                          name='inputs_rnd')
        self.training    = tf.placeholder(tf.bool, [], name='training')

        #-----------------------------------------------------------------------
        # The generator
        #-----------------------------------------------------------------------
        self.alpha = alpha
        with tf.variable_scope('generator'):
            x1 = tf.layers.dense(self.inputs_rnd, 4*4*512)
            x1 = tf.reshape(x1, [-1, 4, 4, 512])
            x1 = tf.layers.batch_normalization(x1, training=self.training)
            x1 = LeakyReLU(x1, self.alpha)
            # 4x4x512 now

            x2 = tf.layers.conv2d_transpose(x1, 256, 5, strides=2,
                                            padding='same')
            x2 = tf.layers.batch_normalization(x2, training=self.training)
            x2 = LeakyReLU(x2, self.alpha)
            # 8x8x256 now

            x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides=2,
                                            padding='same')
            x3 = tf.layers.batch_normalization(x3, training=self.training)
            x3 = LeakyReLU(x3, self.alpha)
            # 16x16x128 now

            # Output layer
            self.gen_logits = tf.layers.conv2d_transpose(x3, 3, 5, strides=2,
                                                         padding='same')
            # 32x32x3 now
            self.gen_out = tf.tanh(self.gen_logits)

    #---------------------------------------------------------------------------
    def __discriminator(self, x, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            x1 = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
            x1 = LeakyReLU(x1, self.alpha)
            # 16x16x64

            x2 = tf.layers.conv2d(x1, 128, 5, strides=2, padding='same')
            x2 = tf.layers.batch_normalization(x2, training=self.training)
            x2 = LeakyReLU(x2, self.alpha)
            # 8x8x128

            x3 = tf.layers.conv2d(x2, 256, 5, strides=2, padding='same')
            x3 = tf.layers.batch_normalization(x3, training=self.training)
            x3 = LeakyReLU(x3, self.alpha)
            # 4x4x256

            # Flatten it
            flat = tf.reshape(x3, (-1, 4*4*256))
            logits = tf.layers.dense(flat, 1)
            out = tf.sigmoid(logits)

        return out, logits

    #---------------------------------------------------------------------------
    def build_discriminator(self, image_size):
        self.inputs_real = tf.placeholder(tf.float32, [None, *image_size],
                                          name='inputs_real')

        #-----------------------------------------------------------------------
        # Process input so that it matches what the generator produces
        #-----------------------------------------------------------------------
        with tf.variable_scope('process_real'):
            processed = self.inputs_real/128-1

        #-----------------------------------------------------------------------
        # Real discriminator
        #-----------------------------------------------------------------------
        ret = self.__discriminator(processed, 'discriminator', False)
        self.dsc_real_out    = ret[0]
        self.dsc_real_logits = ret[1]

        #-----------------------------------------------------------------------
        # Fake discriminator
        #-----------------------------------------------------------------------
        ret = self.__discriminator(self.gen_out, 'discriminator', True)
        self.dsc_fake_out    = ret[0]
        self.dsc_fake_logits = ret[1]

    #---------------------------------------------------------------------------
    def get_optimizers(self, learning_rate=0.0002, beta1=0.2, smooth=0.1):
        #-----------------------------------------------------------------------
        # Define loss functions
        #-----------------------------------------------------------------------
        with tf.variable_scope('loses'):
            dsc_real_loss = tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.dsc_real_logits,
                labels=tf.ones_like(self.dsc_real_logits)))

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

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.variable_scope('optimizers'):
                with tf.variable_scope('deiscriminator_optimizer'):
                    dsc_train_opt = tf.train.AdamOptimizer(learning_rate,
                                                           beta1=beta1) \
                      .minimize(dsc_loss, var_list=dsc_vars)
                with tf.variable_scope('generator_optimizer'):
                    gen_train_opt = tf.train.AdamOptimizer(learning_rate,
                                                           beta1=beta1) \
                      .minimize(gen_loss, var_list=gen_vars)

        return dsc_train_opt, gen_train_opt, dsc_loss, gen_loss
