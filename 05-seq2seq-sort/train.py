#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   06.07.2017
#-------------------------------------------------------------------------------

import argparse
import pickle
import sys

import tensorflow as tf
import numpy as np

from sort_net import SortNet
from tqdm import tqdm

#-------------------------------------------------------------------------------
# Extract vocabulary
#-------------------------------------------------------------------------------
def extract_vocab(seqs):
    standard_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    chars = set([c for seq in seqs for c in seq])
    int_to_word = dict(enumerate(standard_tokens + list(chars)))
    word_to_int = {w: i for i, w in int_to_word.items()}
    return int_to_word, word_to_int

#-------------------------------------------------------------------------------
# Pad sequences
#-------------------------------------------------------------------------------
def pad_sequences(seqs, padding):
    max_len = max([len(s) for s in seqs])
    return [s + [padding] * (max_len - len(s)) for s in seqs]

#-------------------------------------------------------------------------------
# Build a batch generator
#-------------------------------------------------------------------------------
def gen_batches(sources, targets, batch_size, src_padding, tgt_padding):
    blen = batch_size * (len(sources)//batch_size)
    for i in range(0, blen, batch_size):
        sbatch = sources[i:i+batch_size]
        tbatch = targets[i:i+batch_size]
        sbatch = np.array(pad_sequences(sbatch, src_padding))
        tbatch = np.array(pad_sequences(tbatch, tgt_padding))
        slens  = np.array([len(s) for s in sbatch])
        tlens  = np.array([len(t) for t in tbatch])

        yield sbatch, tbatch, slens, tlens

#-------------------------------------------------------------------------------
# Parse the commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train the sorting network')
parser.add_argument('--name', default='test', help='project name')
parser.add_argument('--strings', default='strings.txt', help='strings')
parser.add_argument('--sorted', default='sorted.txt', help='sorted strings')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lstm-size', type=int, default=64, help='lstm size')
parser.add_argument('--lstm-layers', type=int, default=2, help='lstm layers')
parser.add_argument('--enc-embedding-size', type=int, default=15,
                    help='encoding embedding size')
parser.add_argument('--dec-embedding-size', type=int, default=15,
                    help='decoding embedding size')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--tensorboard-dir', default="tb",
                    help='name of the tensorboard data directory')

args = parser.parse_args()

#-------------------------------------------------------------------------------
# Print the parameters
#-------------------------------------------------------------------------------
print('[i] Project name:           ', args.name)
print('[i] Strings file:           ', args.strings)
print('[i] Sorted file:            ', args.sorted)
print('[i] # epochs:               ', args.epochs)
print('[i] Batch size:             ', args.batch_size)
print('[i] LSTM size:              ', args.lstm_size)
print('[i] LSTM layers:            ', args.lstm_layers)
print('[i] Encoding embedding size:', args.enc_embedding_size)
print('[i] Decoding embedding size:', args.dec_embedding_size)
print('[i] Tensorboard directory:  ', args.tensorboard_dir)

#-------------------------------------------------------------------------------
# Read the data
#-------------------------------------------------------------------------------
try:
    with open(args.strings, 'r') as f:
        inputs = f.readlines()
    with open(args.sorted, 'r') as f:
        targets = f.readlines()
except IOError as e:
    print('[!]', str(e))
    sys.exit(1)

inputs = [seq.strip() for seq in inputs]
targets = [seq.strip() for seq in targets]

src_int_to_word, src_word_to_int = extract_vocab(inputs)
tgt_int_to_word, tgt_word_to_int = extract_vocab(targets)

enc_inputs = list(map(lambda x: [src_word_to_int[w] for w in x], inputs))
enc_targets = list(map(lambda x: [tgt_word_to_int[w] for w in x] + \
                                 [tgt_word_to_int['</s>']], targets))

max_length = max(map(lambda x: len(x), enc_inputs))
spad = src_word_to_int['<pad>']
tpad = src_word_to_int['<pad>']

#-------------------------------------------------------------------------------
# Set up the network
#-------------------------------------------------------------------------------
net = SortNet(args.lstm_size, args.lstm_layers,
              len(src_int_to_word), args.enc_embedding_size,
              tgt_word_to_int, args.dec_embedding_size,
              max_length+1)

optimizer, loss = net.get_optimizer(tgt_word_to_int, max_length+1)

training_loss   = tf.placeholder(tf.float32)
validation_loss = tf.placeholder(tf.float32)
tf.summary.scalar('training_loss', training_loss)
tf.summary.scalar('validation_loss', validation_loss)

summary_tensor  = tf.summary.merge_all()

#-------------------------------------------------------------------------------
# Set the session up
#-------------------------------------------------------------------------------
n_batches = len(enc_inputs)//args.batch_size
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
    saver       = tf.train.Saver(max_to_keep=25)
    for e in range(args.epochs):
        #-----------------------------------------------------------------------
        # Training
        #-----------------------------------------------------------------------
        generator   = gen_batches(enc_inputs, enc_targets, args.batch_size,
                                  spad, tpad)
        description = '[i] Epoch {:>2}/{}'.format(e+1, args.epochs)
        vx, vy, vxl, vyl = next(generator)
        generator   = tqdm(generator, total=n_batches-1, desc=description,
                           unit='batches')

        total_loss = 0.
        for x, y, xl, yl in generator:
            feed = {net.inputs:         x,
                    net.targets:        y,
                    net.batch_size:     x.shape[0],
                    net.src_seq_length: xl,
                    net.tgt_seq_length: yl}
            batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed)
            total_loss += batch_loss

        total_loss /= n_batches
        print('[i] Training loss:', total_loss)

        #-----------------------------------------------------------------------
        # Validation
        #-----------------------------------------------------------------------
        feed = {net.inputs:         vx,
                net.targets:        vy,
                net.batch_size:     vx.shape[0],
                net.src_seq_length: vxl,
                net.tgt_seq_length: vyl}
        val_loss = sess.run(loss, feed_dict=feed)
        print('[i] Validation loss:', val_loss)

        #-----------------------------------------------------------------------
        # Write the summary
        #-----------------------------------------------------------------------
        feed = {training_loss:   total_loss,
                validation_loss: val_loss}
        summaries = sess.run(summary_tensor, feed_dict=feed)
        file_writer.add_summary(summaries, e+1)

        #-----------------------------------------------------------------------
        # Make a snapshot
        #-----------------------------------------------------------------------
        if((e+1) % 1 == 0):
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
        data['lstm_size']          = args.lstm_size
        data['lstm_layers']        = args.lstm_layers
        data['src_word_to_int']    = src_word_to_int
        data['src_int_to_word']    = src_int_to_word
        data['tgt_word_to_int']    = tgt_word_to_int
        data['tgt_int_to_word']    = tgt_int_to_word
        data['enc_embedding_size'] = args.enc_embedding_size
        data['dec_embedding_size'] = args.dec_embedding_size
        data['tgt_max_length']     = max_length+1
        pickle.dump(data, f)
except (IOError) as e:
    print(str(e))
    sys.exit(1)

print('[i] All done.')
