#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   26.06.2017
#-------------------------------------------------------------------------------

import argparse
import pickle
import string
import sys

import tensorflow as tf
import numpy as np

from sentiment_rnn import SentimentRNN
from utils import str2bool

#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Test reviews')
parser.add_argument('--name', default='test',
                    help='project name')
parser.add_argument('data', metavar='review_1.txt', type=str, nargs=1,
                    help='review to judge')
parser.add_argument('--checkpoint', type=int, default=-1,
                    help='checkpoint to restore; -1 is the most recent')
parser.add_argument('--embeddings-file', default='embeddings.npy',
                    help='the numpy array with the embeddings')
parser.add_argument("--use-embeddings", type=str2bool, default='y',
                    help="use pre-trained embeddings")
args = parser.parse_args()

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------
state = tf.train.get_checkpoint_state(args.name)
if state is None:
    print('[!] No network state found in ' + args.name)
    sys.exit(1)

try:
    checkpoint_file = state.all_model_checkpoint_paths[args.checkpoint]
except IndexError:
    print('[!] Cannot find checkpoint ' + str(args.checkpoint_file))
    sys.exit(1)

try:
    with open(args.name + '/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
        lstm_size   = metadata['lstm_size']
        lstm_layers = metadata['lstm_layers']
        word_to_int = metadata['word_to_int']

except (IOError) as e:
    print(str(e))
    sys.exit(1)

print('[i] Project name:           ', args.name)
print('[i] Network checkpoint:     ', checkpoint_file)
print('[i] Loading embeddings from:', args.embeddings_file)
print('[i] Using embeddings:       ', args.use_embeddings)
print('[i] Vocabulary size:        ', len(word_to_int))

#-------------------------------------------------------------------------------
# Load embeddings
#-------------------------------------------------------------------------------
embedding = None
if args.use_embeddings:
    print('[i] Loading the embedding...')
    try:
        embedding = np.load(args.embeddings_file)
    except (IOError) as e:
        print(str(e))
        sys.exit(1)

#-------------------------------------------------------------------------------
# Process input
#-------------------------------------------------------------------------------
try:
    with open(args.data[0], 'r') as f:
        review = f.read()
except (IOError) as e:
    print(str(e))
    sys.exit(1)

review = ''.join([c for c in review.lower() if c not in string.punctuation])
review = review.split()
review = review[:200]

encoded = []
for word in review:
    if word in word_to_int:
        encoded.append(word_to_int[word])
    else:
        encoded.append(0)

encoded_np = np.zeros([1, 200], dtype=np.int32)
encoded_np[0, 200-len(encoded):] = encoded

#-------------------------------------------------------------------------------
# Build the network
#-------------------------------------------------------------------------------
net    = SentimentRNN(1, lstm_size, lstm_layers, embedding, len(word_to_int))
saver  = tf.train.Saver()
result = tf.cast(tf.round(net.output), tf.int32)

with tf.Session() as sess:
    saver.restore(sess, checkpoint_file)
    state = sess.run(net.initial_state)
    feed = {net.input:         encoded_np,
            net.keep_prob:     1,
            net.initial_state: state}
    if embedding is not None:
        feed[net.embedding] = embedding
    ret = sess.run(result, feed_dict=feed)
    if ret[0][0] == 1:
        result = 'Positive'
    else:
        result = 'Negative'

print('[i] Verdict:                ', result)
