#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   02.06.2017
# Based on https://github.com/udacity/deep-learning/tree/master/intro-to-rnns
#-------------------------------------------------------------------------------

import argparse
import sys
import pickle
import os

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from character_rnn import CharacterRNN

#-------------------------------------------------------------------------------
def gen_batches(data, n_seqs, n_steps):
    """Create a generator that returns batches of size n_seqs x n_steps."""

    characters_per_batch = n_seqs * n_steps
    n_batches = len(data) // characters_per_batch

    # Keep only enough characters to make full batches
    data = data[:n_batches*characters_per_batch]
    data = data.reshape([n_seqs, -1])

    for n in range(0, data.shape[1], n_steps):
        x = data[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('text', metavar='data.txt', type=str, nargs=1,
                    help='the text body to train on')
parser.add_argument('--name', default='test',
                    help='project name')
parser.add_argument('--lstm-size', type=int, default=512,
                    help='number of LSTM cells in on layer')
parser.add_argument('--lstm-layers', type=int, default=2,
                    help='number of lstm layers')
parser.add_argument('--n-seqs', type=int, default=100,
                    help='number of training sequences')
parser.add_argument('--n-steps', type=int, default=100,
                    help='number of characters in each sequence in a batch')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of training epochs')
parser.add_argument('--tensorboard-dir', default="tb",
                    help='name of the tensorboard data directory')

args = parser.parse_args()

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
print('Project name:             ', args.name)
print('LSTM size:                ', args.lstm_size)
print('LSTM layers:              ', args.lstm_layers)
print('Number of sequences:      ', args.n_seqs)
print('Number of steps:          ', args.n_steps)
print('Number of training epochs:', args.epochs)
print('Tensorboard directory:    ', args.tensorboard_dir)

#-------------------------------------------------------------------------------
# Open the text file and create the vocabulary dictionary
#-------------------------------------------------------------------------------
print('[i] Reading text from {}...'.format(args.text[0]))

try:
    with open(args.text[0], 'r') as f:
        text = f.read()
    vocab = sorted(list(set(text)))
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    print('[i] Creating directory {}...'.format(args.name))
    os.makedirs(args.name)
except (IOError) as e:
    print(str(e))
    sys.exit(1)

encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

#-------------------------------------------------------------------------------
# Build the network
#-------------------------------------------------------------------------------
net = CharacterRNN(len(vocab), args.n_seqs, args.n_steps, args.lstm_size,
                   args.lstm_layers)
loss, optimizer = net.get_training_tensors()

#-------------------------------------------------------------------------------
# Train the network
#-------------------------------------------------------------------------------
saver = tf.train.Saver(max_to_keep=100)
characters_per_batch = args.n_seqs * args.n_steps
n_batches = len(encoded) // characters_per_batch
summary_tensor = tf.summary.merge_all()
iteration      = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
    for e in range(args.epochs):
        new_state   = sess.run(net.initial_state)
        total_loss  = 0
        generator   = gen_batches(encoded, args.n_seqs, args.n_steps)
        description = 'Epoch {:>2}/{}'.format(e+1, args.epochs)
        for x, y in tqdm(generator, total=n_batches, desc=description,
                         unit='batches'):
            feed = {net.inputs:        x,
                    net.targets:       y,
                    net.keep_prob:     0.5,
                    net.initial_state: new_state}
            summary, batch_loss, new_state, _ = sess.run([summary_tensor,
                                                          loss,
                                                          net.final_state,
                                                          optimizer],
                                                         feed_dict=feed)
            total_loss += batch_loss
            file_writer.add_summary(summary, iteration)
            iteration += 1

        print('Training loss: {:.4f}... '.format(total_loss/n_batches))

        if((e+1) % 5 == 0):
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
    with open(args.name + '/metadata.p', 'wb') as f:
        data = {}
        data['lstm_size']    = args.lstm_size
        data['lstm_layers']  = args.lstm_layers
        data['vocab_to_int'] = vocab_to_int
        data['int_to_vocab'] = int_to_vocab
        pickle.dump(data, f)
except (IOError) as e:
    print(str(e))
    sys.exit(1)

print('[i] All done.')
