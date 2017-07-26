#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   26.07.2017
#-------------------------------------------------------------------------------

import argparse
import pickle
import sys
import cv2

import tensorflow as tf
import numpy as np

from dc_gan import DCGAN
from utils import gen_sample_summary

#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate data based on a model')
parser.add_argument('--name', default='test', help='project name')
parser.add_argument('--output-file', default='output.png', help='output file')
parser.add_argument('--checkpoint', type=int, default=-1,
                    help='checkpoint to restore; -1 is the most recent')
args = parser.parse_args()

#-------------------------------------------------------------------------------
# Check if we can load the model
#-------------------------------------------------------------------------------
try:
    with open(args.name + '/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
except (IOError) as e:
    print(str(e))
    sys.exit(1)

state = tf.train.get_checkpoint_state(args.name)
if state is None:
    print('No network state found in ' + args.name)
    sys.exit(1)

try:
    checkpoint = state.all_model_checkpoint_paths[args.checkpoint]
except IndexError:
    print('Cannot find chackpoint ' + str(args.checkpoint))
    sys.exit(1)

random_size = metadata['random_size']

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
print('Project name:        ', args.name)
print('Network checkpoint:  ', checkpoint)
print('Random vector size:  ', random_size)
print('Output file:         ', args.output_file)

#-------------------------------------------------------------------------------
# Create the network
#-------------------------------------------------------------------------------
net = DCGAN(random_size, 0.2)

#-------------------------------------------------------------------------------
# Generate samples and store them to a file
#-------------------------------------------------------------------------------
with tf.Session() as sess:
    saver   = tf.train.Saver()
    saver.restore(sess, checkpoint)
    rnd     = np.random.uniform(-1, 1, size=(15, random_size))
    feed    = {net.inputs_rnd: rnd, net.training: False}
    samples = sess.run(net.gen_out, feed_dict=feed)
    board   = gen_sample_summary(samples)
    cv2.imwrite(args.output_file, cv2.cvtColor(board, cv2.COLOR_RGB2BGR))

print('All done.')
