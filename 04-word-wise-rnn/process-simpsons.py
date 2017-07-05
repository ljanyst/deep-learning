#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   29.06.2017
#-------------------------------------------------------------------------------

import string
import sys
import csv

from collections import namedtuple, defaultdict
from operator import attrgetter

Line = namedtuple('line', ['number', 'text', 'is_dialog'])

#-------------------------------------------------------------------------------
# Process the character name
#-------------------------------------------------------------------------------
def process_name(name):
    name = ''.join([c for c in name if c not in string.punctuation])
    name = name.replace(' ', '_')
    return name

#-------------------------------------------------------------------------------
# Process commandline and read the input file
#-------------------------------------------------------------------------------
if len(sys.argv) < 3:
    print('[i] Usage:', sys.argv[0], 'input output')
    sys.exit(1)

episodes = defaultdict(list)
try:
    with open(sys.argv[1], 'r') as f:
        data = csv.reader(f)
        next(data)
        for line in data:
            episodes[int(line[1])].append(Line(int(line[2]), line[3],
                                               len(line[-1]) != 0))

except (IOError, UnicodeDecodeError) as e:
    print('[!]', str(e))

#-------------------------------------------------------------------------------
# Process the episodes
#-------------------------------------------------------------------------------
for e in episodes:
    episodes[e] = sorted(episodes[e], key=attrgetter('number'))

    new_scene = True
    scenes = []
    scene  = []
    for line in episodes[e]:
        if line.is_dialog:
            new_scene = False
            text = line.text.split(':')
            text = process_name(text[0]) + ':' + ':'.join(text[1:])
            new_line = Line(line.number, text, line.is_dialog)
            scene.append(new_line)
        elif new_scene == False:
            new_scene = True
            scenes.append(scene)
            scene = []
    episodes[e] = scenes

#-------------------------------------------------------------------------------
# Write the output
#-------------------------------------------------------------------------------
try:
    with open(sys.argv[2], 'w') as f:
        for e in episodes:
            for scene in episodes[e]:
                for line in scene:
                    print(line.text, file=f)
                print('', file=f)
except IOError as e:
    print('[!]', str(e))
    sys.exit(1)
