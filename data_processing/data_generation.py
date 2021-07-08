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

def read_grayscale_pngs(path, width=20, height=13):
    # print(len([name for name in os.listdir('{}/.'.format(path)) if os.path.isfile(name)]))
    num_files = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))]) # Calculate amount of files in directory

    ids = np.empty(num_files)
    images = np.empty((num_files, 13, 20))

    for i, image_path in enumerate(glob("{}/*".format(path))):
        id = int(os.path.splitext(os.path.basename(image_path))[0])
        ids[i] = int(id) # File ID: the reason for this is that before many images were eliminated and I want to keep their original number and indexes wouldn't work
        images[i] = np.array(imread(image_path))[:, :, 0] # Pixel data: It's grayscale so take only Red values from [R, G, B, A]
    return ids, images


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
    image[image < 35] = 0        print("Generating mirrored illegal data")

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


import seaborn as sns
if __name__ == '__main__':
    
    legal = read_grayscale_pngs('cleaned/legal')
    illegal = read_grayscale_pngs('cleaned/illegal')

    ids, images = legal
    pbar = tqdm(total=len(images), desc="Legal data")
    for i, image in enumerate(images):

        shifted_images = shift_images(image)
        for j, shifted_image in enumerate(shifted_images):
            plt.imsave('shifted/legal/{}_{}.png'.format(ids[i], j), shifted_image, cmap='gray', vmin=-10, vmax=245)
        
        pbar.update()

    
    ids, images = illegal
    pbar = tqdm(total=len(images), desc="Illegal data")
    for i, image in enumerate(images):

        shifted_images = shift_images(image)
        for j, img in enumerate(shifted_images):
            plt.imsave('shifted/illegal/{}_{}.png'.format(ids[i], j), img, cmap='gray', vmin=-10, vmax=245)

        rotated_image = np.fliplr(image)
        plt.imsave('mirrored/illegal/{}.png'.format(ids[i]), rotated_image, cmap='gray', vmin=-10, vmax=245)
        
        mirrored_images = np.flip(images, axis=2) # Flip (mirror) horizontally
        for j, img in enumerate(mirrored_images):
            plt.imsave('shifted/illegal/{}_{}.png'.format(ids[i], j), img, cmap='gray', vmin=-10, vmax=245)
        
        pbar.update()