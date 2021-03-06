#!/usr/bin/env python

import numpy as np
from scipy import ndimage
from imageio import imread
import matplotlib.pyplot as plt
from sys import argv
import operator
from tqdm import tqdm

from argparse import ArgumentParser
from pathlib import Path


def read_grayscale_pngs(path, width=20, height=13):
    if path is None:
        return None
    if not path.exists():
        print("Path {} doesn't exist".format(path))
        return None

    num_files = len(list(path.glob('*.png'))) # Calculate amount of files in directory
    if num_files == 0:
        print("Path {} doesn't contain any images".format(path))
        return None

    images = np.empty((num_files, 13, 20))

    for i, image_path in enumerate(sorted(path.glob('*.png'), key=lambda f: int(f.stem))):
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

    if np.all(total == 0):
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


def rotate_image(image, angle, mode='nearest'):
    rotated = ndimage.rotate(image, angle, mode=mode)

    orig_average = weighted_average_indexes(image)

    rotated_average = weighted_average_indexes(rotated)
    if orig_average is None or rotated_average is None:
        return None

    # Shift the image so the center is at the same place
    center_diff = tuple(map(operator.sub, orig_average, rotated_average))
    median_value = np.median(rotated)

    shifted = ndimage.shift(rotated, np.array([center_diff[1], center_diff[0]]), cval=median_value)

    shape_diff = tuple(map(operator.sub, rotated.shape, image.shape))

    # Crop the image since it got bigger by rotation, it's centered so cutting the rest is enough
    return shifted[: image.shape[0], : image.shape[1]]


# import seaborn as sns
if __name__ == '__main__':

    # Define the interface
    ap = ArgumentParser(description="Read data created by touchpad capture program, and from illegal data, remove data that don't belong there, such as empty images, or some anomalies")
    ap.add_argument('dest', type=Path, nargs='?', default="out", help="""Destination folder, where to save the data. Inside this folder, another two folders "legal" and "illegal" if needed, are created. Default: "out".""")
    ap.add_argument('--legal', type=Path, metavar="PATH", help="dataset containing finger touches, that the palm ejection algorithm shouldn't reject")
    ap.add_argument('--illegal', type=Path, metavar="PATH", help="dataset containing palm touches, that the palm rejection algorithm should reject")
    ap.add_argument('-r', metavar="DEGREES", nargs='+', type=float, help="don't use rotating for generation", dest='rotate')
    ap.add_argument('-s', action='store_true', help="Use shifting for generation", dest='shift')
    ap.add_argument('-m', action='store_true', help="Use mirroring for generation", dest='mirror')

    args = ap.parse_args()

    if len(argv) == 1:
        ap.print_help()
        quit()

    if not args.rotate and not args.shift and not args.mirror:
        print("No generation option provided. Choose all of them, rotation with -1 and 1 degrees? [y/N]")
        valid = {'yes': True, 'y': True, 'no': False, 'n': False}
        while True:
            choice = input().lower()
            if choice == "" or not valid[choice]:
                quit()
            if valid[choice]:
                args.rotate = [-1.0, 1.0]
                args.shift = True
                args.mirror = True
                break

    if args.rotate:
        if 0.0 in args.rotate:
            args.rotate.remove(0.0)
        for angle in args.rotate:
            if not(angle > -360 and angle < 360):
                print("Rotate angle {} outside the range".format(angle))
                quit()

    legal = read_grayscale_pngs(args.legal)
    illegal = read_grayscale_pngs(args.illegal)

    for key, images in {'legal': legal, 'illegal': illegal}.items():
        if images is None:
            continue

        pbar = tqdm(total=len(images), desc=key)
        dest = args.dest/key

        if args.shift:
            dest_shifted = dest/'shifted'
            dest_shifted.mkdir(parents=True, exist_ok=True)

        if args.mirror:
            dest_mirrored = dest/'mirrored'
            dest_mirrored.mkdir(parents=True, exist_ok=True)

        if args.shift and args.mirror:
            dest_shift_mirrored = dest/'shift_mirrored'
            dest_shift_mirrored.mkdir(parents=True, exist_ok=True)

        if args.rotate:
            dest_rotated = []
            for angle in args.rotate:
                temp_dest = dest/'rotated{}'.format(angle)
                temp_dest.mkdir(parents=True, exist_ok=True)
                dest_rotated.append(temp_dest)
        
        for i, image in enumerate(images):
            if args.shift:
                images_shifted = shift_images(image) 
                for j, img in enumerate(images_shifted):
                    plt.imsave('{}/{}_{}.png'.format(dest_shifted, i, j), img, cmap='gray', vmin=-10, vmax=245)
                    if args.mirror:
                        plt.imsave('{}/{}_{}.png'.format(dest_shift_mirrored, i, j), np.fliplr(img), cmap='gray', vmin=-10, vmax=245)
            if args.mirror:
                plt.imsave('{}/{}.png'.format(dest_mirrored, i), np.fliplr(image), cmap='gray', vmin=-10, vmax=245)
            if args.rotate:
                for angle, temp_dest in zip(args.rotate, dest_rotated):
                    rotated_image = rotate_image(image, angle)
                    if rotated_image is not None:
                        plt.imsave('{}/{}.png'.format(temp_dest, i), rotated_image, cmap='gray', vmin=-10, vmax=245)

            pbar.update()