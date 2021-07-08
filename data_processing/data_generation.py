#!/usr/bin/env python

import numpy as np
from numpy.lib.function_base import median
from scipy import ndimage
from glob import glob
from imageio import imread
import matplotlib.pyplot as plt
import os
import operator
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser


def read_grayscale_pngs(path, width=20, height=13):
    if path is None:
        return None

    # print(len([name for name in os.listdir('{}/.'.format(path)) if os.path.isfile(name)]))
    num_files = len(list(path.glob('*'))) # Calculate amount of files in directory
    # num_files = len([f for f in path.iterdir() if path.joinpath(f).is_file()]) # Calculate amount of files in directory

    ids = np.empty(num_files)
    images = np.empty((num_files, 13, 20))

    for i, image_path in enumerate(path.glob('*.png')):
        # id = int(image_path.stem)
        # ids[i] = int(id) # File ID: the reason for this is that before many images were eliminated and I want to keep their original number and indexes wouldn't work
        images[i] = np.array(imread(image_path))[:, :, 0] # Pixel data: It's grayscale so take only Red values from [R, G, B, A]
    return images

def shift_images(image, width=20, height=13):
    # Get indices for cells with a value higher than 25,
    #  which corresponds to where are significant touches located
    # Flip columns so x axis is first column and y axis second
    # Transpose the array so x and y axes can be easily indexed
    indices = np.fliplr(np.argwhere(image > 25)).T

    if indices.size == 0:
        return None

    # Get edges of values above 25 so a "no-go reactangle" can be created.
    x_min = np.min(indices[0])
    x_max = np.max(indices[0])

    y_min = np.min(indices[1])
    y_max = np.max(indices[1])

    # Create range of movement that will tell which directions can be image 
    #  shifted to generate mroe images 
    range_of_movement = {
        '+x': width - x_max - 2     if x_min > 0 and x_max < width-1    else 0,
        '-x': x_min - 1             if x_min > 0 and x_max < width-1    else 0,
        '+y': height - y_max - 2    if y_min > 0 and y_max < height-1   else 0,
        '-y': y_min - 1             if y_min > 0 and y_max < height-1   else 0,
    }

    # Generate an array corresponding to every value in the range
    x = list(range(-1, -range_of_movement['-x']-1, -1)) + list(range(range_of_movement['+x']+1))
    y = list(range(-1, -range_of_movement['-y']-1, -1)) + list(range(range_of_movement['+y']+1))

    # Create tuples from the values
    arr = np.dstack(np.meshgrid(x, y)).reshape(-1, 2) # Make a cartesian product of those two arrays
    if len(arr) == 1:
        return None
    arr = np.delete(arr, np.where((arr == [0, 0]).all(axis=1)), axis=0) # Remove [0, 0] since it doesn't 

    median_value = np.median(image)

    for movement in arr:
        yield ndimage.shift(image, np.array([movement[1], movement[0]]), cval=median_value)


def weighted_average_indexes(image):
    # Values below 35 are not substantial and they mess with the average so replace them with zero
    image[image < 35] = 0
    total = np.sum(image) 

    if total == 0:
        return None

    val = 0
    for index, xsum in enumerate(np.sum(image, axis=0)):
        val += index * xsum
    xpos = val / total 

    val = 0
    for index, ysum in enumerate(np.sum(image, axis=1)):
        val += index * ysum
    ypos = val / total

    return xpos, ypos


def rotate_image(image, angle):
    rotated = ndimage.rotate(image, angle, mode='nearest')

    # Shift the image so the center is at the same place
    center_diff = tuple(map(operator.sub, weighted_average_indexes(image), weighted_average_indexes(rotated)))
    median_value = np.median(rotated)

    shifted = ndimage.shift(rotated, np.array([center_diff[1], center_diff[0]]), cval=median_value)

    shape_diff = tuple(map(operator.sub, rotated.shape, image.shape))

    # Crop the image since it got bigger by rotation
    return shifted[0:-shape_diff[0], 0:-shape_diff[1]]


# import seaborn as sns
if __name__ == '__main__':

    # Define the interface
    ap = ArgumentParser(description="Read data created by touchpad capture program, and from illegal data, remove data that don't belong there, such as empty images, or some anomalies")
    ap.add_argument('dest', type=Path, nargs='?', default="out", help="""Destination folder, where to save the data. Inside this folder, another two folders "legal" and "illegal" if needed, are created. Default: "out".""")
    ap.add_argument('--legal', type=Path, metavar="PATH", help="dataset containing finger touches, that the palm ejection algorithm shouldn't reject")
    ap.add_argument('--illegal', type=Path, metavar="PATH", help="dataset containing palm touched, that the palm rejection algorithm should reject")
    ap.add_argument('-W', type=int, default=20, metavar="WIDTH", help="width of each image")
    ap.add_argument('-H', type=int, default=13, metavar="HEIGHT", help="height of each image")

    # args = ap.parse_args()
    args = ap.parse_args(['--legal', 'out/legal/orig', '--illegal', 'out/illegal/orig'])

    # if len(argv) == 1:
    #     ap.print_help()
    #     quit()
    
    legal = read_grayscale_pngs(args.legal)
    illegal = read_grayscale_pngs(args.illegal)

    if legal is not None:
        pbar = tqdm(total=len(legal), desc="Legal data")
        for i, image in enumerate(legal):
            dest = args.dest/'legal'
            images_shifted = shift_images(image)
            dest_shifted = dest/'shifted'
            dest_shifted.mkdir(parents=True, exist_ok=True)
            for j, img in enumerate(images_shifted):
                plt.imsave('{}/{}_{}.png'.format(dest_shifted, i, j), img, cmap='gray', vmin=-10, vmax=245)

            mirrored_images = np.flip(legal, axis=2) # Flip (mirror) horizontally
            dest_mirrored = dest/'mirrored'
            dest_mirrored.mkdir(parents=True, exist_ok=True)
            for j, img in enumerate(mirrored_images):
                plt.imsave('{}/{}_{}.png'.format(dest_mirrored, i, j), img, cmap='gray', vmin=-10, vmax=245)
            
            pbar.update()

    
    if illegal is not None:
        pbar = tqdm(total=len(illegal), desc="Illegal data")
        for i, image in enumerate(illegal):

            images_shifted = shift_images(image)
            dest_shifted = dest/'shifted'
            dest_shifted.mkdir(parents=True, exist_ok=True)
            for j, img in enumerate(images_shifted):
                plt.imsave('{}/{}_{}.png'.format(dest_shifted, i, j), img, cmap='gray', vmin=-10, vmax=245)

            rotated_image = np.fliplr(image)
            plt.imsave('{}/{}.png'.format(dest_shifted, i), rotated_image, cmap='gray', vmin=-10, vmax=245)
            
            mirrored_images = np.flip(illegal, axis=2) # Flip (mirror) horizontally
            dest_mirrored = dest/'mirrored'
            dest_mirrored.mkdir(parents=True, exist_ok=True)
            for j, img in enumerate(mirrored_images):
                plt.imsave('{}/{}_{}.png'.format(dest_mirrored, i, j), img, cmap='gray', vmin=-10, vmax=245)
            
            pbar.update()