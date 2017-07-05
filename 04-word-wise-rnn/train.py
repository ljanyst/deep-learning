#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   29.06.2017
#-------------------------------------------------------------------------------

import argparse
import pickle

import tensorflow as tf
import numpy as np

from collections import Counter
from word_rnn import WordRNN
from tqdm import tqdm

#-------------------------------------------------------------------------------
def encode_punctuation(text):
    """Replace punctuation with corresponding tokens"""
    text = text.replace('.',  ' <PERIOD> ')
    text = text.replace(',',  ' <COMMA> ')
    text = text.replace('"',  ' <QUOTATION_MARK> ')
    text = text.replace(';',  ' <SEMICOLON> ')
    text = text.replace('!',  ' <EXCLAMATION_MARK> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace('(',  ' <LEFT_PAREN> ')
    text = text.replace(')',  ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':',  ' <COLON> ')
    return text

#-------------------------------------------------------------------------------
def tokenize_text(text):
    """Split the text into tokens"""
    return encode_punctuation(text).split()

#-------------------------------------------------------------------------------
def gen_batches(text, n_seqs, n_steps):
    """
    Generate sequence batches
    """
    out_text = text[1:]
    out_text.append(text[0])

    words_per_batch = n_seqs * n_steps
    n_batches       = len(text)//words_per_batch
    text            = text[:n_batches*words_per_batch]
    text            = np.array(text)
    text            = text.reshape([n_seqs, -1])
    out_text        = out_text[:n_batches*words_per_batch]
    out_text        = np.array(out_text)
    out_text        = out_text.reshape([n_seqs, -1])

    for i in range(0, text.shape[1], n_steps):
        yield text[:,i:i+n_steps], out_text[:,i:i+n_steps]

#-------------------------------------------------------------------------------
# Parse the commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train a word-wise RNN')
parser.add_argument('text', metavar='data.txt', type=str, nargs=1,
                    help='the text body to train on')
parser.add_argument('--name', default='test', help='project name')
parser.add_argument('--lstm-size', type=int, default=2048,
                    help='number of LSTM cells in on layer')
parser.add_argument('--lstm-layers', type=int, default=3,
                    help='number of lstm layers')
parser.add_argument('--n-seqs', type=int, default=50,
                    help='number of training sequences')
parser.add_argument('--n-steps', type=int, default=15,
                    help='number of characters in each sequence in a batch')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of training epochs')
parser.add_argument('--tensorboard-dir', default="tb",
                    help='name of the tensorboard data directory')
parser.add_argument('--embedding-size', type=int, default=200,
                    help='embedding size')

args = parser.parse_args()

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
print('[i] Project name:          ', args.name)
print('[i] Input:                 ', args.text[0])
print('[i] LSTM size:             ', args.lstm_size)
print('[i] LSTM layers:           ', args.lstm_layers)
print('[i] # of sequences:        ', args.n_seqs)
print('[i] # of steps:            ', args.n_steps)
print('[i] # of training epochs:  ', args.epochs)
print('[i] Embedding size:        ', args.embedding_size)
print('[i] Tensorboard directory: ', args.tensorboard_dir)

#-------------------------------------------------------------------------------
# Process the input text
#-------------------------------------------------------------------------------
try:
    with open(args.text[0]) as f:
        data = f.read()
except IOError as e:
    print('[!]', str(e))
    sys.exit(1)

data = tokenize_text(data.lower())
cnt = Counter(data)
vocab_size = len(cnt)
print('[i] Vocabulary size:       ', vocab_size)

vocab_sorted = sorted(cnt, key=cnt.get, reverse=True)
int_to_word = dict(enumerate(vocab_sorted))
word_to_int = {w: i for i, w in enumerate(vocab_sorted)}
data_encoded = [word_to_int[w] for w in data]

#-------------------------------------------------------------------------------
# Set up the network
#-------------------------------------------------------------------------------
net = WordRNN(vocab_size, args.embedding_size, args.lstm_size, args.lstm_layers)
optimizer, loss = net.get_optimizer()

training_loss    = tf.placeholder(tf.float32)
training_loss_op = tf.summary.scalar('training_loss', training_loss)

words_per_batch = args.n_seqs * args.n_steps
n_batches       = len(data_encoded)//words_per_batch

#-------------------------------------------------------------------------------
# Set the session up
#-------------------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
    saver       = tf.train.Saver(max_to_keep=25)
    for e in range(args.epochs):
        #-----------------------------------------------------------------------
        # Training
        #-----------------------------------------------------------------------
        state       = sess.run(net.initial_state,
                               feed_dict={net.n_seqs: args.n_seqs})
        generator   = gen_batches(data_encoded, args.n_seqs, args.n_steps)
        description = '[i] Epoch {:>2}/{}'.format(e+1, args.epochs)
        generator   = tqdm(generator, total=n_batches, desc=description,
                           unit='batches')

        total_loss = 0.
        for x, y in generator:
            feed = {net.inputs:        x,
                    net.targets:       y,
                    net.keep_prob:     0.5,
                    net.n_seqs:        args.n_seqs,
                    net.initial_state: state}
            batch_loss, state, _ = sess.run([loss, net.final_state, optimizer],
                                            feed_dict=feed)
            total_loss += batch_loss

        total_loss /= n_batches
        print('[i] Training loss:', total_loss)

        #-----------------------------------------------------------------------
        # Write the summary
        #-----------------------------------------------------------------------
        feed = {training_loss: total_loss}
        summary = sess.run(training_loss_op, feed_dict=feed)
        file_writer.add_summary(summary, e+1)

        #-----------------------------------------------------------------------
        # Make a snapshot
        #-----------------------------------------------------------------------
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
        data['lstm_size']      = args.lstm_size
        data['lstm_layers']    = args.lstm_layers
        data['word_to_int']    = word_to_int
        data['int_to_word']    = int_to_word
        data['embedding_size'] = args.embedding_size
        data['n_steps']        = args.n_steps
        pickle.dump(data, f)
except (IOError) as e:
    print(str(e))
    sys.exit(1)

print('[i] All done.')
