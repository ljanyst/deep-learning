#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   06.07.2017
#-------------------------------------------------------------------------------

import argparse
import string
import random

#-------------------------------------------------------------------------------
# Parse the commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate random strings')
parser.add_argument('--strings', default='strings.txt',
                    help='generated strings')
parser.add_argument('--sorted', default='sorted.txt',
                    help='sorted strings')
parser.add_argument('--number', type=int, default=10000000,
                    help='number of strings to generate')
parser.add_argument('--max-length', type=int, default=10,
                    help='maximum length of a string')

args = parser.parse_args()

#-------------------------------------------------------------------------------
# Print the parameters
#-------------------------------------------------------------------------------
print('[i] Strings file:',             args.strings)
print('[i] Sorted file: ',             args.sorted)
print('[i] # of strings to generate:', args.number)
print('[i] Maximum length:          ', args.max_length)

#-------------------------------------------------------------------------------
# Generate the data
#-------------------------------------------------------------------------------
print('[i] Generating data...')
try:
    with open(args.strings, 'w') as strs:
        with open(args.sorted, 'w') as sstrs:
            for i in range(args.number):
                l = random.randrange(2, args.max_length)
                s = random.sample(string.ascii_lowercase, l)
                r = sorted(s)
                print(''.join(s), file=strs)
                print(''.join(r), file=sstrs)
except IOError as e:
    print('[!]', str(e))
    sys.exit(1)

print('[i] All done.')
