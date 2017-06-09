#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   09.06.2017
#-------------------------------------------------------------------------------

import readline
import argparse
import pickle
import sys

import tensorflow as tf
import numpy as np

#-------------------------------------------------------------------------------
class EmbeddingHelper:
    """Find nearest vectors in an embedding using cosine distance"""

    #---------------------------------------------------------------------------
    def __init__(self, embedding):
        self.sess         = tf.Session()
        self.inputs       = tf.placeholder(tf.float32,
                                           [None, embedding.shape[1]],
                                           name='inputs')
        self.test_vec     = tf.placeholder(tf.float32, [1, embedding.shape[1]],
                                           name='test_vec')
        self.cos_distance = tf.matmul(self.inputs, tf.transpose(self.test_vec))

        #-----------------------------------------------------------------------
        # Compute normalized embedding matrix
        #-----------------------------------------------------------------------
        row_sum    = tf.reduce_sum(tf.square(self.inputs), axis=1,
                                   keep_dims=True)
        norm       = tf.sqrt(row_sum)
        self.normalized = self.inputs / norm
        self.embedding = self.sess.run(self.normalized,
                                       feed_dict={self.inputs: embedding})

    #---------------------------------------------------------------------------
    def __enter__(self):
        self.sess.__enter__()
        return self

    #---------------------------------------------------------------------------
    def __exit__(self, type, value, traceback):
        self.sess.__exit__(type, value, traceback)

    #---------------------------------------------------------------------------
    def find_nearest(self, vec, n=3):
        vec      = np.array([vec], dtype=np.float32)
        norm_vec = self.sess.run(self.normalized, feed_dict={self.inputs: vec})
        dist     = self.sess.run(self.cos_distance,
                                 feed_dict={self.inputs:   self.embedding,
                                            self.test_vec: norm_vec})
        dist = np.reshape(dist, self.embedding.shape[0])
        nearest = (-dist).argsort()[:n+2]
        ret = []
        for i in nearest:
            ret.append((i, dist[i]))
        return ret

#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Test embeddings')
parser.add_argument('data', metavar='embeddings.npy', type=str, nargs=1,
                    help='the numpy array with the embeddings')
parser.add_argument('--metadata-file', default='',
                    help='file to load the metadata from')

args = parser.parse_args()

metadata_file = args.metadata_file
if not metadata_file:
    metadata_file = args.data[0].split('.')
    metadata_file[-1] = 'pkl'
    metadata_file = '.'.join(metadata_file)

print('[i] Loading embeddings from:', args.data[0])
print('[i] Loading metadata from:  ', metadata_file)

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------
try:
    embeddings = np.load(args.data[0])
    with open(metadata_file, 'rb') as f:
        tokens = pickle.load(f)
except (IOError) as e:
    print(str(e))
    sys.exit(1)

word_to_int = {w: i for i, w in enumerate(tokens)}
int_to_word = {i: w for i, w in enumerate(tokens)}

print('[i] Number of tokens:       ', len(tokens))
print('[i] Vector size:            ', embeddings.shape[1])

#-------------------------------------------------------------------------------
class LookupError(Exception):
    pass

#-------------------------------------------------------------------------------
def parse_line(line, embeddings, word_to_int):
    accept_word = True
    op          = '+'
    line = line.split()
    vec  = np.zeros(embeddings.shape[1], dtype=np.float32)
    for token in line:
        if accept_word:
            if not token in word_to_int:
                raise LookupError('Word {} not found'.format(token))
            token_vec = embeddings[word_to_int[token],:]
            if op == '+':
                vec += token_vec
            else:
                vec -= token_vec
            accept_word = False
        else:
            if token == '+' or token == '-':
                op = token
            else:
                raise LookupError('Unknown operation: {}'.format(token))
            accept_word = True
    return vec

#-------------------------------------------------------------------------------
# Test the embeddings
#-------------------------------------------------------------------------------
with EmbeddingHelper(embeddings) as eh:
    while True:
        try:
            line = input('[i] Prompt (CTRL-D to quit): ')
            vec = parse_line(line, embeddings, word_to_int)
            print('[i] Looking for words nearest to {}...'.format(line))
            nearest = eh.find_nearest(vec, 5)
            for index, similarity in nearest:
                print('[i]    {} ({:0.5f})'.format(int_to_word[index], similarity))
        except (EOFError) as e:
            break
        except (LookupError) as e:
            print('[!] Lookup error:', str(e))
