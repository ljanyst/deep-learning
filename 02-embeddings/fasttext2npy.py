#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   09.06.2017
#-------------------------------------------------------------------------------

import argparse
import sys
import pickle
import numpy as np
from tqdm import tqdm

#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Convert fasttext to numpy')
parser.add_argument('data', metavar='data.vec', type=str, nargs=1,
                    help='data in the vec format')
parser.add_argument('--embedding-file', default='',
                    help='file name for the embedding')
parser.add_argument('--metadata-file', default='',
                    help='file name for the metadata')
args = parser.parse_args()

embedding = args.embedding_file
metadata  = args.metadata_file

if not embedding:
    embedding = args.data[0].split('.')
    embedding[-1] = 'npy'
    embedding = '.'.join(embedding)

if not metadata:
    metadata = args.data[0].split('.')
    metadata[-1] = 'pkl'
    metadata = '.'.join(metadata)

print('[i] Input file:        ', args.data[0])
print('[i] Numpy output:      ', embedding)
print('[i] Metadata output:   ', metadata)

#-------------------------------------------------------------------------------
# Process the input
#-------------------------------------------------------------------------------
try:
    with open(args.data[0], 'r') as f:
        header  = f.readline().split()
        ntokens = int(header[0])
        vecsize = int(header[1])
        print('[i] Number of tokens:  ', ntokens)
        print('[i] Vector size:       ', vecsize)
        arr    = np.zeros((ntokens, vecsize), dtype=np.float32)
        tokens = []
        i      = 0
        for line in tqdm(f, total=ntokens, unit='vectors',
                         desc='[i] Processing vectors'):
            l = line.split()
            if len(l) == vecsize:
                tokens.append(' ')
            else:
                token = ' '.join(l[:-vecsize])
                tokens.append(token)
                l = l[-vecsize:]
            l = [float(x) for x in l]
            l = np.array(l)
            arr[i,:] = l
            i += 1
except (IOError) as e:
    print(str(e))
    sys.exit(1)

#-------------------------------------------------------------------------------
# Save data
#-------------------------------------------------------------------------------
print('[i] Saving embedding...')
np.save(embedding, arr)

print('[i] Saving metadata...')
try:
    with open(metadata, 'wb') as f:
        pickle.dump(tokens, f)
except (IOError) as e:
    print(str(e))
    sys.exit(1)

print('[i] All done.')
