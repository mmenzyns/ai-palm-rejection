import numpy as np
from numpy.lib.function_base import median
from scipy import ndimage
from glob import glob
from imageio import imread
import matplotlib.pyplot as plt
import os


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

def get_movement_range(image, width=20, height=13):
    # Get indices for cells wit             h a value higher than 25,
    #  which corresponds to where are significant touches
    # Flip columns so x axis is first column and y axis second
    # Transpose the array so x and y axes can be easily indexed
    indices = np.fliplr(np.argwhere(image > 25)).T

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

    return range_of_movement


def generate_from_movement_range(image, range_of_movement):
    # Create arrays for range of movements

    x = list(range(-1, -range_of_movement['-x']-1, -1)) + list(range(range_of_movement['+x']+1))
    y = list(range(-1, -range_of_movement['-y']-1, -1)) + list(range(range_of_movement['+y']+1))

    arr = np.dstack(np.meshgrid(x, y)).reshape(-1, 2) # Make a cartesian product of those two arrays
    if len(arr) == 1:
        return None
    arr = np.delete(arr, np.where((arr == [0, 0]).all(axis=1)), axis=0) # Remove [0, 0] since it doesn't 

    median_value = np.median(image)

    for movement in arr:
        yield ndimage.shift(image, np.array([movement[1], movement[0]]), cval=median_value)


if __name__ == '__main__':
    legal = read_grayscale_pngs('out/legal')
    illegal = read_grayscale_pngs('out/illegal')

    for ids, images in legal:
        for i, image in enumerate(images):
            shift_range = get_movement_range(image)
            shifted_images = generate_from_movement_range(image, shift_range)
            for j, shifted_image in enumerate(shifted_images):
                plt.imsave('shifted/legal/{}_{}.png'.format(ids[i], j), shifted_image, cmap='gray', vmin=-10, vmax=245)


    
    for ids, images in illegal:
        for i, image in enumerate(images):
            shift_range = get_movement_range(image)
            shifted_images = generate_from_movement_range(image, shift_range)
            for j, img in enumerate(shifted_images):
                plt.imsave('shifted/illegal/{}_{}.png'.format(ids[i], j), img, cmap='gray', vmin=-10, vmax=245)

            rotated_image = np.fliplr(image)
            plt.imsave('mirrored/illegal/{}.png'.format(ids[i]), rotated_image, cmap='gray', vmin=-10, vmax=245)
            
            mirrored_images = np.flip(images, axis=2) # Flip (mirror) horizontally
            for j, img in enumerate(mirrored_images):
                plt.imsave('shifted/illegal/{}_{}.png'.format(ids[i], j), img, cmap='gray', vmin=-10, vmax=245)