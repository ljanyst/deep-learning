#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   09.06.2017
#-------------------------------------------------------------------------------

import argparse
import pickle
import sys

import numpy as np

from tqdm import tqdm

#-------------------------------------------------------------------------------
# Parse the commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Extract a sumbset of embeddings')
parser.add_argument('data', metavar='data.npy', type=str, nargs=2,
                    help='file to load the embeddings from')
parser.add_argument('--extract-metadata-file', default='',
                    help='file name to load the metadata from')
parser.add_argument('--extract-cap', type=int, default=100000,
                    help='number of vectors to extract from source')
parser.add_argument('--metadata-file', default='',
                    help='file name to load the metadata from')
parser.add_argument('--target-metadata-file', default='',
                    help='file name for the target metadata')
args = parser.parse_args()

metadata_file = args.metadata_file
if not metadata_file:
    metadata_file = args.data[0].split('.')
    metadata_file[-1] = 'pkl'
    metadata_file = '.'.join(metadata_file)

target_metadata_file = args.target_metadata_file
if not target_metadata_file:
    target_metadata_file = args.data[1].split('.')
    target_metadata_file[-1] = 'pkl'
    target_metadata_file = '.'.join(target_metadata_file)

print('[i] Loading embeddings from:      ', args.data[0])
print('[i] Loading metadata from:        ', metadata_file)

if args.extract_metadata_file:
    print('[i] Loading extract metadata from:', args.extract_metadata_file)
else:
    print('[i] Extraction cap:               ', args.extract_cap)
print('[i] Storing target metadata to:   ', target_metadata_file)
print('[i] Storing target embeddings to: ', args.data[1])

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------
try:
    embeddings = np.load(args.data[0])
    with open(metadata_file, 'rb') as f:
        tokens = pickle.load(f)
    if args.extract_metadata_file:
        with open(args.extract_metadata_file, 'rb') as f:
            extract_tokens = pickle.load(f)
except (IOError) as e:
    print(str(e))
    sys.exit(1)

word_to_int = {w: i for i, w in enumerate(tokens)}

target_tokens = []
if args.extract_metadata_file:
    for token in extract_tokens:
        if token in word_to_int:
            target_tokens.append(token)
else:
    target_tokens = tokens[:args.extract_cap]

print('[i] Number of tokens:             ', len(tokens))
if args.extract_metadata_file:
    print('[i] Number of tokens to extract:  ', len(extract_tokens))
print('[i] Number of target tokens:      ', len(target_tokens))
print('[i] Vector size:                  ', embeddings.shape[1])

#-------------------------------------------------------------------------------
# Extract the data
#-------------------------------------------------------------------------------
arr = np.zeros((len(target_tokens), embeddings.shape[1]), dtype=np.float32)
i   = 0

for token in tqdm(target_tokens, unit='tokens', desc='[i] Processing'):
    arr[i,:] = embeddings[word_to_int[token],:]
    i += 1

print('[i] Saving new embeddings...')
np.save(args.data[1], arr)

print('[i] Saving new metadata...')
try:
    with open(target_metadata_file, 'wb') as f:
        pickle.dump(target_tokens, f)
except (IOError) as e:
    print(str(e))
    sys.exit(1)

print('[i] All done.')
