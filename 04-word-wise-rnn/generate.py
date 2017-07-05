#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   29.06.2017
#-------------------------------------------------------------------------------

import argparse
import pickle
import sys

import tensorflow as tf
import numpy as np

from word_rnn import WordRNN

#-------------------------------------------------------------------------------
def pick_word(probabilities, int_to_word):
    """
    Pick the next word according to the probability distribution
    :return: String of the predicted word
    """
    c = np.random.choice(len(probabilities), 1, p=probabilities)[0]
    return int_to_word[c]

#-------------------------------------------------------------------------------
def decode_punctuation(text):
    """Replace the punctuation tokens with actual punctuation"""
    text = text.replace('<PERIOD>',           '.')
    text = text.replace('<COMMA>',            ',')
    text = text.replace('<QUOTATION_MARK>',   '"')
    text = text.replace('<SEMICOLON>',        ';')
    text = text.replace('<EXCLAMATION_MARK>', '! ')
    text = text.replace('<QUESTION_MARK>',    '? ')
    text = text.replace('<LEFT_PAREN>',       '(')
    text = text.replace('<RIGHT_PAREN>',      ')')
    text = text.replace('<HYPHENS>',          '--')
    text = text.replace('<NEW_LINE>',         '\n')
    text = text.replace('<COLON>',            ':')
    text = text.replace(' .', '.')
    text = text.replace(' ,', ',')
    text = text.replace(' ;', ';')
    text = text.replace(' !', '! ')
    text = text.replace(' ?', '? ')
    text = text.replace('( ', '(')
    text = text.replace(' )', ')')
    text = text.replace(' :', ':')
    text = text.split('\n')
    text = [word.strip() for word in text]
    text = '\n'.join(text)
    return text

#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate data based on a model')
parser.add_argument('--name', default='test',
                    help='project name')
parser.add_argument('--prime', default='the',
                    help='some words to prime the network with')
parser.add_argument('--samples', type=int, default=500,
                    help='number of characters go generate')
parser.add_argument('--checkpoint', type=int, default=-1,
                    help='checkpoint to restore; -1 is the most recent')

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

lstm_size      = metadata['lstm_size']
lstm_layers    = metadata['lstm_layers']
word_to_int    = metadata['word_to_int']
int_to_word    = metadata['int_to_word']
embedding_size = metadata['embedding_size']
n_steps        = metadata['n_steps']

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
print('Project name:        ', args.name)
print('Network checkpoint:  ', checkpoint)
print('LSTM size:           ', lstm_size)
print('LSTM layers:         ', lstm_layers)
print('Embedding size:      ', embedding_size)
print('Priming text:        ', args.prime)
print('Samples to generate: ', args.samples)

#-------------------------------------------------------------------------------
# Generate samples
#-------------------------------------------------------------------------------
net = WordRNN(len(word_to_int), embedding_size, lstm_size, lstm_layers)
saver = tf.train.Saver()

with tf.Session() as sess:
    print('[i] Restoring a checkpoint from', checkpoint)
    saver.restore(sess, checkpoint)
    state = sess.run(net.initial_state, feed_dict={net.n_seqs: 1})

    seq = [args.prime]
    for _ in range(args.samples):
        x = [[word_to_int[w] for w in seq[-n_steps:]]]
        feed = {net.inputs:        x,
                net.keep_prob:     1,
                net.n_seqs:        1,
                net.initial_state: state}
        probs, state = sess.run([net.output, net.final_state],
                                feed_dict=feed)
        seq.append(pick_word(probs[0][-1], int_to_word))

#-------------------------------------------------------------------------------
# Process tokens
#-------------------------------------------------------------------------------
text = ' '.join(seq)
print(decode_punctuation(text))
