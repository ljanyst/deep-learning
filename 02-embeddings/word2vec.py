#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   04.06.2017
# Based on https://github.com/udacity/deep-learning/tree/master/embeddings
#-------------------------------------------------------------------------------

import argparse
import random
import math
import sys
import os

import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
from collections import Counter
from tqdm import tqdm

#-------------------------------------------------------------------------------
def process_punctuation(text):
    """Replace punctuation with corresponding tokens"""
    text = text.lower()
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
    return process_punctuation(text).split()

#-------------------------------------------------------------------------------
class SkipGram:
    #---------------------------------------------------------------------------
    def __init__(self, vocab_len, embedding_len):
        #-----------------------------------------------------------------------
        # Placeholders for input and targets
        #-----------------------------------------------------------------------
        self.inputs    = tf.placeholder(tf.int32, [None], name="inputs")
        self.targets   = tf.placeholder(tf.int32, [None, 1], name="targets")

        #-----------------------------------------------------------------------
        # The embedding.
        # We use the embedding lookup instead of matrix multiplication for
        # performance reasons. Since it's a huge matrix and only one of the
        # input vector elements is non-zero, it's faster to just select the row
        # in the matrix instead of doing a huge number of multiplications with
        # the same end result.
        #-----------------------------------------------------------------------
        with tf.name_scope('embedding'):
            self.embedding = tf.Variable(tf.random_uniform(
                                           (vocab_len, embedding_len), -1, 1),
                                         name = 'word_embedding')
            tf.summary.histogram('word_embedding', self.embedding)
            embed = tf.nn.embedding_lookup(self.embedding, self.inputs)

        #-----------------------------------------------------------------------
        # Loss.
        # We'll use the so called negative sampling
        # If your target vocabulary (number of classes) is really big, it is
        # very hard to use regular softmax, because you have to calculate
        # probability for every word in dictionary. By using
        # sampled_softmax_loss you only take in account subset V of your
        # vocabulary to calculate your loss.
        # Paper: http://arxiv.org/pdf/1412.2007v2.pdf
        #-----------------------------------------------------------------------
        with tf.name_scope('loss'):
            softmax_w = tf.Variable(tf.truncated_normal(
                                      (vocab_len, embedding_len), stddev=0.1),
                                    name="softmax_w")
            softmax_b = tf.Variable(tf.zeros(vocab_len), name="softmax_b")

            with tf.name_scope('sampled_softmax_loss'):
                loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b,
                                                  self.targets, embed, 100,
                                                  vocab_len)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss', self.loss)

        #-----------------------------------------------------------------------
        # The optimizer
        #-----------------------------------------------------------------------
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    #---------------------------------------------------------------------------
    def setup_projection(self, logdir, vocab, counter):
        #-----------------------------------------------------------------------
        # Set up the projector config
        #-----------------------------------------------------------------------
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.embedding.name
        embedding.metadata_path = logdir + '/metadata.tsv'

        self.summary_writer = tf.summary.FileWriter(logdir)
        projector.visualize_embeddings(self.summary_writer, config)

        self.saver  = tf.train.Saver()
        self.logdir = logdir

        #-----------------------------------------------------------------------
        # Write the TSV metadata file
        #-----------------------------------------------------------------------
        with open(embedding.metadata_path, "w") as f:
            f.write('Word\tFrequency\n')
            for word in vocab:
                f.write('{}\t{}\n'.format(word, counter[word]))

        return self.summary_writer

    #---------------------------------------------------------------------------
    def save_checkpoint(self, session, step):
        self.saver.save(session, self.logdir + "/model.ckpt", step)

#-------------------------------------------------------------------------------
def get_target(words, idx, window_size=5):
    """Get a list of words in a window around an index."""
    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = set(words[start:idx] + words[idx+1:stop+1])

    return list(target_words)

#-------------------------------------------------------------------------------
def gen_batches(words, batch_size, window_size=5):
    """Generate batches of words and targets"""

    n_batches = len(words)//batch_size
    if len(words) % batch_size != 0:
        n_batches += 1

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield np.array(x), np.array(y)

#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Create embeddings from text')
parser.add_argument('text', metavar='data.txt', type=str, nargs=1,
                    help='the text body to train on')
parser.add_argument('--name', default='test',
                    help='project name')
parser.add_argument('--embedding-size', type=int, default=300,
                    help='size of the embedding')
parser.add_argument('--min-frequency', type=int, default=5,
                    help='minimum word frequency in the corpus')
parser.add_argument('--batch-size', type=int, default=1000,
                    help='batch size')
parser.add_argument('--window-size', type=int, default=10,
                    help='word window size')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of training epochs')
parser.add_argument('--tensorboard-dir', default="tb",
                    help='name of the tensorboard data directory')
parser.add_argument('--output-file', default="embeddings.npy",
                    help='file to save the weights to')

args = parser.parse_args()

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
print('[i] Project name:             ', args.name)
print('[i] Embedding size:           ', args.embedding_size)
print('[i] Minimum word frequency:   ', args.min_frequency)
print('[i] Batch size:               ', args.batch_size)
print('[i] Window size:              ', args.window_size)
print('[i] Number of training epochs:', args.epochs)
print('[i] Tensorboard directory:    ', args.tensorboard_dir)
print('[i] Output file:              ', args.output_file)

#-------------------------------------------------------------------------------
# Open the text file, tokenize the text body, remove rare words, and subsample
# frequent words
#-------------------------------------------------------------------------------
print('[i] Reading text from {}...'.format(args.text[0]))

try:
    with open(args.text[0], 'r') as f:
        text = f.read()
    print('[i] Creating directory {}...'.format(args.name))
    os.makedirs(args.name)
except (IOError) as e:
    print(str(e))
    sys.exit(1)

print('[i] Tokenizing...')
text = tokenize_text(text)
print('[i] Number of tokens:', len(text))
cnt = Counter(text)
print('[i] Vocabulary size:', len(cnt))

print('[i] Replacing rare words with <UNSEEN>...')
text = [word if cnt[word] > args.min_frequency else '<UNSEEN>' for word in text]
cnt = Counter(text)

print('[i] Subsampling frequent words...')
t          = 1e-5
num_tokens = float(len(text))
freqs      = {word: count/num_tokens for word, count in cnt.items()}
drop_prob  = {word: (1-math.sqrt(t/freqs[word])) for word in cnt}
text       = [word for word in text if random.random() > drop_prob[word]]
print('[i] Found {} usable tokens'.format(len(text)))

print('[i] Creating vocabulary dictionary...')
cnt = Counter(text)
sorted_vocab = sorted(cnt, key=cnt.get, reverse=True)
int_to_word  = {ind: word for ind, word in enumerate(sorted_vocab)}
word_to_int  = {word: ind for ind, word in enumerate(sorted_vocab)}
vocab_size   = len(word_to_int)
encoded_text = [word_to_int[word] for word in text]
print('[i] Vocabulary size:', vocab_size)

#-------------------------------------------------------------------------------
# Train the network
#-------------------------------------------------------------------------------
print('[i] Setting up the network...')
net = SkipGram(vocab_size, args.embedding_size)
net.setup_projection(args.tensorboard_dir, sorted_vocab, cnt)

saver = tf.train.Saver(max_to_keep=5)
n_batches = len(encoded_text)//args.batch_size
if len(encoded_text) % args.batch_size != 0:
    n_batches += 1

summary_tensor = tf.summary.merge_all()

with tf.Session() as sess:
    iteration = 1
    sess.run(tf.global_variables_initializer())
    net.summary_writer.add_graph(sess.graph)
    for e in range(args.epochs):
        loss   = 0
        length = 0
        generator = gen_batches(encoded_text, args.batch_size,
                                args.window_size)
        description = '[i] Epoch {:>2}/{}'.format(e+1, args.epochs)
        for x, y in tqdm(generator, total=n_batches, desc=description,
                         unit='batches'):

            feed = {net.inputs:  x,
                    net.targets: y.reshape([-1, 1])}

            if iteration % 1000 != 0:
                train_loss, _ = sess.run([net.loss, net.optimizer],
                                         feed_dict=feed)
            else:
                summary, train_loss, _ = sess.run([summary_tensor,
                                                   net.loss,
                                                   net.optimizer],
                                                  feed_dict=feed)

                net.save_checkpoint(sess, iteration)
                net.summary_writer.add_summary(summary, iteration)

            loss   += train_loss*len(x)
            length += len(x)
            iteration += 1

        loss /= float(length)
        print('[i] Training loss: {:.4f}... '.format(loss))
        if (e+1) % 5 == 0:
            checkpoint = '{}/e{}.ckpt'.format(args.name, e+1)
            saver.save(sess, checkpoint)
            print('[i] Checkpoint saved:', checkpoint)
    checkpoint = '{}/final.ckpt'.format(args.name)
    saver.save(sess, checkpoint)
    print('[i] Checkpoint saved:', checkpoint)
    embedding = sess.run(net.embedding)
    np.save(args.output_file, embedding)
    print('[i] Data saved:', args.output_file)
