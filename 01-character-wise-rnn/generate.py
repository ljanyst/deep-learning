#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   02.06.2017
# Based on https://github.com/udacity/deep-learning/tree/master/intro-to-rnns
#-------------------------------------------------------------------------------

import argparse
import pickle
import sys

import tensorflow as tf
import numpy as np

from character_rnn import CharacterRNN

#-------------------------------------------------------------------------------
# Sample predictions using the distribution returned by the network
#-------------------------------------------------------------------------------
def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate data based on a model')
parser.add_argument('--name', default='test',
                    help='project name')
parser.add_argument('--prime', default='The',
                    help='some words to prime the network with')
parser.add_argument('--samples', type=int, default=500,
                    help='number of characters go generate')
parser.add_argument('--checkpoint', type=int, default=-1,
                    help='checkpoint to restore; -1 is the most recent')
parser.add_argument('--top-predictions', type=int, default=5,
                    help='how many of the best predictions to use')

args = parser.parse_args()

#-------------------------------------------------------------------------------
# Check if we can load the model
#-------------------------------------------------------------------------------
try:
    with open(args.name + '/metadata.p', 'rb') as f:
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

lstm_size    = metadata['lstm_size']
lstm_layers  = metadata['lstm_layers']
vocab_to_int = metadata['vocab_to_int']
int_to_vocab = metadata['int_to_vocab']

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
print('Project name:                    ', args.name)
print('Network checkpoint:              ', checkpoint)
print('LSTM size:                       ', lstm_size)
print('LSTM layers:                     ', lstm_layers)
print('Priming text:                    ', args.prime)
print('Samples to generate:             ', args.samples)
print('Number of top preductions to use:', args.top_predictions)

#-------------------------------------------------------------------------------
# Generate samples
#-------------------------------------------------------------------------------
samples = [c for c in args.prime]
net = CharacterRNN(len(int_to_vocab), 1, 1, lstm_size, lstm_layers)
saver = tf.train.Saver()

with tf.Session() as sess:
    print('[i] Restoring a checkpoint from', checkpoint)
    saver.restore(sess, checkpoint)
    new_state = sess.run(net.initial_state)
    for c in args.prime:
        x = np.zeros((1, 1))
        x[0,0] = vocab_to_int[c]
        feed = {net.inputs: x,
                net.keep_prob: 1.,
                net.initial_state: new_state}
        preds, new_state = sess.run([net.outputs, net.final_state],
                                    feed_dict=feed)

    c = pick_top_n(preds, len(vocab_to_int), args.top_predictions)
    samples.append(int_to_vocab[c])

    for i in range(args.samples):
        x[0,0] = c
        feed = {net.inputs: x,
                net.keep_prob: 1.,
                net.initial_state: new_state}
        preds, new_state = sess.run([net.outputs, net.final_state],
                                    feed_dict=feed)
        c = pick_top_n(preds, len(vocab_to_int), args.top_predictions)
        samples.append(int_to_vocab[c])

print(''.join(samples))
