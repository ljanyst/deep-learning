#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   16.07.2017
#-------------------------------------------------------------------------------

import argparse
import pickle
import sys

import tensorflow as tf
import numpy as np

from sort_net import SortNet

#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate data based on a model')
parser.add_argument('--name', default='test',
                    help='project name')
parser.add_argument('--seq', default='hello',
                    help='sequence of characters to sort')
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

lstm_size          = metadata['lstm_size']
lstm_layers        = metadata['lstm_layers']
src_word_to_int    = metadata['src_word_to_int']
src_int_to_word    = metadata['src_int_to_word']
enc_embedding_size = metadata['enc_embedding_size']
tgt_word_to_int    = metadata['tgt_word_to_int']
tgt_int_to_word    = metadata['tgt_int_to_word']
dec_embedding_size = metadata['dec_embedding_size']
tgt_max_length     = metadata['tgt_max_length']

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
print('[i] Project name:        ', args.name)
print('[i] Network checkpoint:  ', checkpoint)
print('[i] LSTM size:           ', lstm_size)
print('[i] LSTM layers:         ', lstm_layers)
print('[i] Max sequence length: ', tgt_max_length)
print('[i] Sequence to sort:    ', args.seq)

enc_seq = [src_word_to_int[w] for w in args.seq]
src_len = tgt_max_length - 1
enc_seq += [src_word_to_int['<pad>']] * (src_len-len(enc_seq))

#-------------------------------------------------------------------------------
# Generate samples
#-------------------------------------------------------------------------------
net = SortNet(lstm_size, lstm_layers,
              len(src_int_to_word), enc_embedding_size,
              tgt_word_to_int, dec_embedding_size,
              tgt_max_length)
saver = tf.train.Saver()

with tf.Session() as sess:
    print('[i] Restoring a checkpoint from', checkpoint)
    saver.restore(sess, checkpoint)
    
    feed = {net.inputs:         np.array([enc_seq]),
            net.batch_size:     1,
            net.src_seq_length: np.array([len(enc_seq)]),
            net.tgt_seq_length: np.array([len(enc_seq)+1])}
    result = sess.run(net.outputs, feed_dict=feed)[0]

result = ''.join([tgt_int_to_word[i] for i in result[:-1]])

print('[i] Sorted sequence:     ', result)
