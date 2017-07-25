#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   25.07.2017
#-------------------------------------------------------------------------------

import argparse
import pickle

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from mnist_gan import MNISTGAN
from utils import gen_sample_summary
from tqdm import tqdm

#-------------------------------------------------------------------------------
# Batch generator
#-------------------------------------------------------------------------------
def gen_batches(batch_size):
    for _ in range(mnist.train.num_examples//batch_size):
        yield mnist.train.next_batch(batch_size)

#-------------------------------------------------------------------------------
# Parse the commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train a GAN')
parser.add_argument('--name', default='test', help='project name')
parser.add_argument('--tensorboard-dir', default="tb",
                    help='name of the tensorboard data directory')
parser.add_argument('--random-size', type=int, default=128,
                    help='size of the random input to the generator')
parser.add_argument('--hidden-size', type=int, default=256,
                    help='size of the hidden layers')
parser.add_argument('--batch-size', type=int, default=128,
                    help='number of characters in each sequence in a batch')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of training epochs')

args = parser.parse_args()

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
print('[i] Project name:          ', args.name)
print('[i] Random vector size:    ', args.random_size)
print('[i] Hidden layer size:     ', args.hidden_size)
print('[i] Batch size:            ', args.batch_size)
print('[i] # of training epochs:  ', args.epochs)
print('[i] Tensorboard directory: ', args.tensorboard_dir)

#-------------------------------------------------------------------------------
# Download and unpack the MNIST data
#-------------------------------------------------------------------------------
print('[i] Downloading the MNIST data...')
mnist = input_data.read_data_sets('MNIST_data')

#-------------------------------------------------------------------------------
# Create the network
#-------------------------------------------------------------------------------
net = MNISTGAN(args.random_size, args.hidden_size, 784, 0.01)
net.build_discriminator(784, args.hidden_size)
dsc_opt, gen_opt, dsc_loss_op, gen_loss_op = net.get_optimizers()

generator_loss        = tf.placeholder(tf.float32)
generator_loss_op     = tf.summary.scalar('generator_loss', generator_loss)
discriminator_loss    = tf.placeholder(tf.float32)
discriminator_loss_op = tf.summary.scalar('discriminator_loss',
                                          discriminator_loss)

loss_ops              = [generator_loss_op, discriminator_loss_op]

sample_img    = tf.placeholder(tf.uint8, shape=[None, None, None, 3])
sample_img_op = tf.summary.image('sample_img', sample_img)

#-------------------------------------------------------------------------------
# Set the session up
#-------------------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_batches   = mnist.train.num_examples//args.batch_size
    file_writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
    saver       = tf.train.Saver(max_to_keep=10)

    for e in range(args.epochs):

        #-----------------------------------------------------------------------
        # Training
        #-----------------------------------------------------------------------
        generator   = gen_batches(args.batch_size)
        description = '[i] Epoch {:>2}/{}'.format(e+1, args.epochs)
        generator   = tqdm(generator, total=n_batches, desc=description,
                           unit='batches')

        total_dsc_loss = 0
        total_gen_loss = 0
        for x, y in generator:
            rnd = np.random.uniform(-1, 1,
                                    size=(args.batch_size, args.random_size))

            feed = {
                net.inputs_rnd: rnd,
                net.inputs_real: x,
                }
            tensors = [dsc_loss_op, gen_loss_op]
            dsc_loss, gen_loss = sess.run(tensors, feed_dict=feed)
            sess.run(dsc_opt, feed_dict=feed)
            sess.run(gen_opt, feed_dict=feed)
            total_dsc_loss += dsc_loss
            total_gen_loss += gen_loss

        total_dsc_loss /= n_batches
        total_gen_loss /= n_batches

        #-----------------------------------------------------------------------
        # Write summary of the losses
        #-----------------------------------------------------------------------
        feed = {
            discriminator_loss: total_dsc_loss,
            generator_loss:     total_gen_loss}
        summary = sess.run(loss_ops, feed_dict=feed)
        file_writer.add_summary(summary[0], e+1)
        file_writer.add_summary(summary[1], e+1)

        #-----------------------------------------------------------------------
        # Generate image summary
        #-----------------------------------------------------------------------
        if (e+1) % 5 == 0:
            rnd     = np.random.uniform(-1, 1, size=(15, args.random_size))
            feed    = {net.inputs_rnd: rnd}
            samples = sess.run(net.gen_out, feed_dict=feed)
            board   = gen_sample_summary(samples)
            board   = board[np.newaxis,:,:,:]
            summary = sess.run(sample_img_op, feed_dict={sample_img: board})
            file_writer.add_summary(summary, e+1)

        #-----------------------------------------------------------------------
        # Save a checktpoint
        #-----------------------------------------------------------------------
        if (e+1) % 5 == 0:
            checkpoint = '{}/e{}.ckpt'.format(args.name, e+1)
            saver.save(sess, checkpoint)
            print('Checkpoint saved:', checkpoint)

    checkpoint = '{}/final.ckpt'.format(args.name)
    saver.save(sess, checkpoint)
    print('Checkpoint saved:', checkpoint)

#-------------------------------------------------------------------------------
# Write the metadata
#-------------------------------------------------------------------------------
try:
    with open(args.name + '/metadata.pkl', 'wb') as f:
        data = {}
        data['random_size'] = args.random_size
        data['hidden_size'] = args.hidden_size
        pickle.dump(data, f)
except (IOError) as e:
    print(str(e))
    sys.exit(1)

print('[i] All done.')
