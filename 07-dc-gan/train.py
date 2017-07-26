#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   26.07.2017
#-------------------------------------------------------------------------------

import argparse
import scipy.io
import pickle
import os

import tensorflow as tf
import numpy as np

from urllib.request import urlretrieve
from dc_gan import DCGAN
from utils import gen_sample_summary
from tqdm import tqdm

#-------------------------------------------------------------------------------
# Progress bar hook
#-------------------------------------------------------------------------------
class DLProgress(tqdm):
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

#-------------------------------------------------------------------------------
# Batch generator
#-------------------------------------------------------------------------------
def gen_batches(data, batch_size):
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx]

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i+batch_size]

#-------------------------------------------------------------------------------
# Parse the commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train a GAN')
parser.add_argument('--name', default='test', help='project name')
parser.add_argument('--tensorboard-dir', default="tb",
                    help='name of the tensorboard data directory')
parser.add_argument('--batch-size', type=int, default=128,
                    help='number of batches')
parser.add_argument('--random-size', type=int, default=128,
                    help='size of the random input to the generator')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of training epochs')
parser.add_argument('--data-dir', default='data',
                    help='data directory')
args = parser.parse_args()

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
print('[i] Project name:         ', args.name)
print('[i] Batch size:           ', args.batch_size)
print('[i] # of training epochs: ', args.epochs)
print('[i] Data directory:       ', args.data_dir)
print('[i] Tensorboard directory:', args.tensorboard_dir)

#-------------------------------------------------------------------------------
# Download the image data
#-------------------------------------------------------------------------------
if not os.path.isdir(args.data_dir):
    os.makedirs(args.data_dir)

if not os.path.isfile(args.data_dir+"/train_32x32.mat"):
    with DLProgress(unit='B', unit_scale=True, miniters=1,
                    desc='SVHN Training Set') as pbar:
        urlretrieve(
            'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
            args.data_dir+'/train_32x32.mat',
            pbar.hook)

if not os.path.isfile(args.data_dir+"/test_32x32.mat"):
    with DLProgress(unit='B', unit_scale=True, miniters=1,
                    desc='SVHN Testing Set') as pbar:
        urlretrieve(
            'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
            args.data_dir+'/test_32x32.mat',
            pbar.hook)

#-------------------------------------------------------------------------------
# Load and process the data
#-------------------------------------------------------------------------------
print('[i] Process data...')
train = scipy.io.loadmat(args.data_dir+'/train_32x32.mat')['X']
test  = scipy.io.loadmat(args.data_dir+'/test_32x32.mat')['X']
train = np.concatenate([np.rollaxis(train, 3), np.rollaxis(test, 3)], axis=0)

#-------------------------------------------------------------------------------
# Create the network
#-------------------------------------------------------------------------------
net = DCGAN(args.random_size, 0.2)
net.build_discriminator([32, 32, 3])
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
    n_batches   = len(train)//args.batch_size+1
    file_writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
    saver       = tf.train.Saver(max_to_keep=10)

    for e in range(args.epochs):

        #-----------------------------------------------------------------------
        # Training
        #-----------------------------------------------------------------------
        generator   = gen_batches(train, args.batch_size)
        description = '[i] Epoch {:>2}/{}'.format(e+1, args.epochs)
        generator   = tqdm(generator, total=n_batches, desc=description,
                           unit='batches')

        total_dsc_loss = 0
        total_gen_loss = 0
        for x in generator:
            rnd = np.random.uniform(-1, 1,
                                    size=(args.batch_size, args.random_size))

            feed = {
                net.inputs_rnd:  rnd,
                net.inputs_real: x,
                net.training:    False
                }
            tensors = [dsc_loss_op, gen_loss_op]
            dsc_loss, gen_loss = sess.run(tensors, feed_dict=feed)
            feed[net.training] = True
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
        rnd     = np.random.uniform(-1, 1, size=(15, args.random_size))
        feed    = {net.inputs_rnd: rnd, net.training: False}
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
        pickle.dump(data, f)
except (IOError) as e:
    print(str(e))
    sys.exit(1)

print('[i] All done.')
