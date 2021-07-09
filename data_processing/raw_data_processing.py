#!/usr/bin/env python 

from argparse import ArgumentParser
from pathlib import Path
from operator import xor
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sys import argv


def read_files(path, width, height):
    if not path:
        return None

    if path.is_dir():
        data = []
        for filename in path.glob('*'):
            data.extend(load_file(filename, width, height))
        return data
    else:
        return load_file(path, width, height)

def load_file(filename, width, height):
    reader = csv.reader(open(filename, 'r'), delimiter='\t')

    data = []
    for row in reader:
        data.extend(row)

    if len(data) % (13*20) != 0:
        print("Ammount of values in file {} doesn't match width {} and height {}".format(filename, width, height))
        quit()
    return pd.to_numeric(data, errors='ignore', downcast='integer').reshape((-1, height, width))


# Define the interface
ap = ArgumentParser(description="Read images created by touchpad capture program, and from illegal data, remove data that don't belong there, such as empty images, or some anomalies")
ap.add_argument('dest', type=Path, nargs='?', default="out", help="""Destination folder, where to save the data. Inside this folder, another two folders "legal" and "illegal" if needed, are created. Default: "out".""")
ap.add_argument('--legal', type=Path, metavar="PATH", help="dataset containing finger touches, that the palm ejection algorithm shouldn't reject")
ap.add_argument('--illegal', type=Path, metavar="PATH", help="dataset containing palm touched, that the palm rejection algorithm should reject")
ap.add_argument('--nontouch', type=Path, metavar="PATH", help="Remove data from illegal dataset which are similar to this dataset")
ap.add_argument('-W', type=int, default=20, metavar="WIDTH", help="width of each image")
ap.add_argument('-H', type=int, default=13, metavar="HEIGHT", help="height of each image")

args = ap.parse_args(['--illegal','../touchpad_capture/real_data/illegal', '--nontouch', '../touchpad_capture/real_data/nontouches'])
# args = ap.parse_args()

# Comment out if using manual input
# if len(argv) == 1:
#     ap.print_help()
#     quit()

# Check user input
if args.legal is None and args.illegal is None:
    print("Nothing to process (--legal and --illegal unspecified)")
    quit()

for path in (args.legal, args.illegal, args.nontouch):
    if path and not path.exists():
        print("Path \"{}\" doesn't exist".format(path))
        quit()

if args.illegal and not args.nontouch:
    print("Nontouch path unspecified. Proced without incorrect data elimination? [y/N]:")
    valid = {'yes': True, 'y': True, 'no': False, 'n': False}

    while True: # Check if user really doesn't want to specify nontouch dataset
        choice = input().lower()
        if choice == "" or not valid[choice]:
            quit()
        if valid[choice]:
            break

if xor(bool(args.H), bool(args.W)): # Check if one is specified and other is not
    print("Both height and width have to be specified")
    quit()

# Load Data
legal = read_files(args.legal, args.W, args.H)
illegal = read_files(args.illegal, args.W, args.H)
nontouch = read_files(args.nontouch, args.W, args.H)

if illegal is not None:
    dest = args.dest/'illegal'/'orig'
    dest.mkdir(parents=True, exist_ok=True)
    glob = dest.glob('*.png')
    if len(list(glob)) != 0:
        last_file = max((fn.name for fn in glob), key=lambda fn: int(fn).stem) # If there are already files, continue indexes
        base_index = int(last_file.split('.')[0])+1 # Extract a number of the file and increment
    else:
        base_index = 0

    if nontouch is not None:
        # Use feature learning to remove unfit data
        illegal_features = pd.DataFrame({
            'vars': np.var(illegal, axis=(1,2)), 
            'max_values': np.max(illegal, axis=(1,2))
        })
        nontouch_features = pd.DataFrame({
            'vars': np.var(nontouch, axis=(1,2)), 
            'max_values': np.max(nontouch, axis=(1,2))
        })

        # Train Gaussian Mixture Model
        bgm = BayesianGaussianMixture().fit(nontouch_features)

        # Calculate logarithmic likelihoods
        illegal_scores = pd.Series(bgm.score_samples(illegal_features))

        # Eliminate high scores 
        illegal_scores = illegal_scores[illegal_scores < -500]

        for index in illegal_scores.index:
            plt.imsave('{}/{}.png'.format(dest, index + base_index), illegal[index], cmap='gray', vmin=-10, vmax=245)
    else:
        # Save all illegal data
        for index, image in enumerate(illegal, start=base_index):
            plt.imsave('{}/{}.png'.format(dest, index), image, cmap='gray', vmin=-10, vmax=245)

if legal is not None:
    dest = args.dest/'legal'/'orig'
    dest.mkdir(parents=True, exist_ok=True)
    glob = dest.glob('*.png')
    if len(list(glob)) != 0:
        last_file = max((fn.name for fn in glob), key=lambda fn: int(fn).stem) # If there are already files, continue indexes
        base_index = int(last_file.split('.')[0])+1 # Extract a number of the file and increment
    else:
        base_index = 0

    for index, image in enumerate(legal, start=base_index):
        plt.imsave('{}/{}.png'.format(dest, index), image, cmap='gray', vmin=-10, vmax=245)
