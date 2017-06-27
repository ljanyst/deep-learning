#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   26.06.2017
#-------------------------------------------------------------------------------

import argparse
import string
import pickle
import math
import sys

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sentiment_rnn import SentimentRNN
from collections import Counter
from utils import str2bool
from tqdm import tqdm

#-------------------------------------------------------------------------------
# Run the data set through the network
#-------------------------------------------------------------------------------
def process_data(sess, net, features, labels, embedding, generator, batch_size,
                 keep_prob, tensors):
    n_batches  = math.ceil(len(features)/batch_size)
    total_loss = 0
    total_acc  = 0
    state      = sess.run(net.initial_state)
    for i in generator:
        batch_x = features[i:i+args.batch_size, :]
        batch_y = labels[i:i+args.batch_size]
        feed = {net.input:         batch_x,
                net.target:        batch_y,
                net.keep_prob:     keep_prob,
                net.initial_state: state}
        if embedding is not None:
            feed[net.embedding] = embedding

        ret = sess.run(tensors[:2] + [net.final_state] + tensors[2:],
                       feed_dict=feed)
        total_loss += ret[0]
        total_acc  += ret[1]
        state       = ret[2]
    total_loss /= n_batches
    total_acc  /= n_batches
    return total_loss, total_acc

#-------------------------------------------------------------------------------
# Parse the commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train a sentiment analyser')
parser.add_argument('--reviews-file', default='reviews.txt',
                    help='file name to load the reviews from')
parser.add_argument('--labels-file', default='labels.txt',
                    help='file name to load the labels from')
parser.add_argument('--embeddings-file', default='embeddings.npy',
                    help='file to load the embeddings from')
parser.add_argument("--use-embeddings", type=str2bool, default='y',
                    help="use pre-trained embeddings")
parser.add_argument('--metadata-file', default='',
                    help='file name to load the metadata from')
parser.add_argument('--name', default='test',
                    help='project name')
parser.add_argument('--tensorboard-dir', default="tb",
                    help='name of the tensorboard data directory')
parser.add_argument('--lstm-size', type=int, default=256,
                    help='number of LSTM cells in on layer')
parser.add_argument('--lstm-layers', type=int, default=1,
                    help='number of lstm layers')
parser.add_argument('--batch-size', type=int, default=500,
                    help='training batch size')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of training epochs')

args = parser.parse_args()

metadata_file = args.metadata_file
if not metadata_file:
    metadata_file = args.embeddings_file.split('.')
    metadata_file[-1] = 'pkl'
    metadata_file = '.'.join(metadata_file)

#-------------------------------------------------------------------------------
# Print the run information
#-------------------------------------------------------------------------------
print('[i] Project name:             ', args.name)
print('[i] Using embeddings:         ', args.use_embeddings)
if(args.use_embeddings):
    print('[i] Loading embeddings from:  ', args.embeddings_file)
    print('[i] Loading metadata from:    ', metadata_file)
print('[i] Loading reviews from:     ', args.reviews_file)
print('[i] Loading labels from:      ', args.labels_file)
print('[i] LSTM size:                ', args.lstm_size)
print('[i] LSTM layers:              ', args.lstm_layers)
print('[i] Batch size:               ', args.batch_size)
print('[i] # of training epochs:     ', args.epochs)
print('[i] Tensorboard directory:    ', args.tensorboard_dir)

#-------------------------------------------------------------------------------
# Load the input data
#-------------------------------------------------------------------------------
try:
    with open(args.reviews_file, 'r') as f:
        reviews = f.read()
    with open(args.labels_file, 'r') as f:
        labels = f.read()
except (IOError) as e:
    print('[!]', str(e))
    sys.exit(1)

#-------------------------------------------------------------------------------
# Preprocess the input data
#-------------------------------------------------------------------------------
print('[i] Preprocessing data...')
reviews = ''.join([c for c in reviews if c not in string.punctuation])
reviews = reviews.split('\n')
reviews = [r.split() for r in reviews]
labels  = labels.split('\n')

embedding = None
if args.use_embeddings:
    print('[i] Loading the embedding...')
    try:
        embedding = np.load(args.embeddings_file)
        with open(metadata_file, 'rb') as f:
            tokens = pickle.load(f)
    except (IOError) as e:
        print(str(e))
        sys.exit(1)
else:
    print('[i] Creating new embedding...')
    vocab = [word for review in reviews for word in review]
    cnt   = Counter(vocab)
    print('[i] Vocabulary size:          ', len(cnt))
    tokens = sorted(cnt, key=cnt.get, reverse=True)

#-------------------------------------------------------------------------------
# Encode the data
#-------------------------------------------------------------------------------
word_to_int = {w: i for i, w in enumerate(tokens)}
print('[i] Number of reviews:        ', len(reviews))
reviews_encoded = []
labels_encoded  = []
unseen_words    = 0
total_words     = 0
for i in range(len(reviews)):
    review = reviews[i]
    label  = labels[i]
    encoded = []
    for word in review:
        total_words += 1
        if word in word_to_int:
            encoded.append(word_to_int[word])
        else:
            encoded.append(0)
            unseen_words += 1

    if not len(encoded):
        continue

    if label == 'positive':
        labels_encoded.append(1)
    else:
        labels_encoded.append(0)
    reviews_encoded.append(encoded)

print('[i] Number of encoded reviews:', len(reviews_encoded))
print('[i] Number of unseen words:    {}/{} ({:.0f}%)'.format(
  unseen_words, total_words, float(unseen_words)/total_words*100))

#-------------------------------------------------------------------------------
# Truncate or pad with zeros, so that each review is 200 characters long;
# Create training, validation, and test vectors.
#-------------------------------------------------------------------------------
print('[i] Creating feature vectors...')
features = np.zeros((len(reviews_encoded), 200), dtype=np.int32)
labels   = np.zeros((len(reviews_encoded), 1), dtype=np.int32)
for i in range(len(reviews_encoded)):
    r = reviews_encoded[i]
    if len(r) > 200:
        features[i, :] = r[:200]
    else:
        features[i, (200-len(r)):] = r
    labels[i][0] = labels_encoded[i]

features, labels = shuffle(features, labels)
features_train, features, labels_train, labels = train_test_split(
    features, labels, test_size=0.2)
features_valid, features_test, labels_valid, labels_test = train_test_split(
    features, labels, test_size=0.5)

print('[i] # training samples:       ', len(features_train))
print('[i] # validation samples:     ', len(features_valid))
print('[i] # testing samples:        ', len(features_test))

#-------------------------------------------------------------------------------
# Create the network
#-------------------------------------------------------------------------------
net = SentimentRNN(args.batch_size, args.lstm_size, args.lstm_layers,
                   embedding, len(tokens))

with tf.variable_scope('accuracy'):
    correct_pred    = tf.equal(tf.cast(tf.round(net.output), tf.int32), net.target)
    accuracy        = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

training_loss   = tf.placeholder(tf.float32)
training_acc    = tf.placeholder(tf.float32)
validation_loss = tf.placeholder(tf.float32)
validation_acc  = tf.placeholder(tf.float32)
tf.summary.scalar('training_loss',   training_loss)
tf.summary.scalar('training_acc',    training_acc)
tf.summary.scalar('validation_loss', validation_loss)
tf.summary.scalar('validation_acc',  validation_acc)

summary_tensor  = tf.summary.merge_all()

optimizer, loss = net.get_optimizer()

n_training_batches   = math.ceil(len(features_train)/args.batch_size)
n_validation_batches = math.ceil(len(features_valid)/args.batch_size)
n_test_batches = math.ceil(len(features_test)/args.batch_size)

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
        features_train, labels_train = shuffle(features_train, labels_train)
        generator = range(0, len(features_train), args.batch_size)
        description = '[i] Epoch {:>2}/{}'.format(e+1, args.epochs)
        generator = tqdm(generator, total=n_training_batches, desc=description,
                         unit='batches')
        total_training_loss, total_training_acc = process_data(sess, net,
            features_train, labels_train, embedding, generator, args.batch_size,
            0.5, [loss, accuracy, optimizer])

        #-----------------------------------------------------------------------
        # Validation
        #-----------------------------------------------------------------------
        generator = range(0, len(features_valid), args.batch_size)
        total_validation_loss, total_validation_acc = process_data(sess, net,
            features_valid, labels_valid, embedding, generator, args.batch_size,
            1, [loss, accuracy])

        #-----------------------------------------------------------------------
        # Store summaries
        #-----------------------------------------------------------------------
        feed = {training_loss:   total_training_loss,
                training_acc:    total_training_acc,
                validation_loss: total_validation_loss,
                validation_acc:  total_validation_acc}
        summaries = sess.run(summary_tensor, feed_dict=feed)
        file_writer.add_summary(summaries, e+1)

        #-----------------------------------------------------------------------
        # Make a checkpoint
        #-----------------------------------------------------------------------
        if((e+1) % 5 == 0):
            checkpoint = '{}/e{}.ckpt'.format(args.name, e+1)
            saver.save(sess, checkpoint)
            print('[i] Checkpoint saved:', checkpoint)

    checkpoint = '{}/final.ckpt'.format(args.name)
    saver.save(sess, checkpoint)
    print('[i] Checkpoint saved:', checkpoint)

    #---------------------------------------------------------------------------
    # Test
    #---------------------------------------------------------------------------
    generator = range(0, len(features_test), args.batch_size)
    total_test_loss, total_test_acc = process_data(sess, net,
        features_test, labels_test, embedding, generator, args.batch_size,
        1, [loss, accuracy])
    print('[i] Test loss:     {:.03f}'.format(total_test_loss))
    print('[i] Test accuracy: {:.03f}'.format(total_test_acc))

#-------------------------------------------------------------------------------
# Write the metadata
#-------------------------------------------------------------------------------
try:
    with open(args.name + '/metadata.pkl', 'wb') as f:
        data = {}
        data['lstm_size']   = args.lstm_size
        data['lstm_layers'] = args.lstm_layers
        data['word_to_int'] = word_to_int
        pickle.dump(data, f)
except (IOError) as e:
    print('[!]', str(e))
    sys.exit(1)

print('[i] All done.')
