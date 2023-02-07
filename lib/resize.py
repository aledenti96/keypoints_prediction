import cv2
import math
import numpy as np
from random import randint as rand
import albumentations as A

def resize(image,keypoints,out_shape):

    #find factor scaling
    image_shape = image.shape
    # print("Image original shape:",image_shape)
    min_shape = min(image_shape[0],image_shape[1])

    # Images with with smaller dimension between 1000 and 2000 are not resized
    # Others images needs to be downscaled or upscaled.
    # If an image need upscaling, factor_scaling will be > 1 viceversa it will be < 1
    if(min_shape > 2000 or min_shape < 1000):
        # using round_up we prevent images with dimension 999, remain the same because of bad round by int() function
        factor_scaling = round_up((out_shape/min_shape),1)
        width = int(image_shape[1] * factor_scaling)
        height = int(image_shape[0] * factor_scaling)
        dim = (width,height)

        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        image = np.expand_dims(image,2)
        # print("Image expanded:",image)
        # print("Image expanded reshaped:",image.shape)
        keypoints = [(i[0]*factor_scaling,i[1]*factor_scaling) for i in keypoints]


    return image, keypoints

# Round up numbers at a given decimals
def round_up(n, decimals=0):

    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

# def crop(image,keypoints,out_shape):

def random_boundaries(left,right,top,bottom,img_shape,out_shape):

    # identification of freedom degrees to crop right and left image border
    east_ovest_bounds = (max(0,(right - out_shape)),min((img_shape[0] - out_shape),left))
    
    x_min = rand(east_ovest_bounds[0],east_ovest_bounds[1])
    x_max = x_min + out_shape

    # identification of freedom degrees to crop top and bottom image border
    north_south_bounds = (max(0,(top - out_shape)),min((img_shape[1] - out_shape),bottom))

    y_min = rand(north_south_bounds[0],north_south_bounds[1])
    y_max = y_min + out_shape

    return x_max,x_min,y_max,y_min

def crop(x_min,x_max,y_min,y_max,image,keypoints):

    pipeline_crop = A.Compose(
        [A.Crop(x_min,y_min,x_max,y_max)],
        keypoint_params=A.KeypointParams(format='xy')
    )

    cropped = pipeline_crop(image = image,keypoints = keypoints)

    if(len(cropped['keypoints']) < 4):
        print("Original keypoints:",keypoints)
        print("Left crop:",x_min)
        print("Right crop:",x_max)
        print("Bottom crop:", y_min)
        print("Top crop:", y_max)

    return cropped['image'],cropped['keypoints']

def check_boundaries(x_min,x_max,y_min,y_max,img_shape):

    if(x_min < 0 or y_min < 0 or x_max > img_shape[0] or y_max > img_shape[1]):
        print("Error")
        print("x_min:",x_min)
        print("x_max:",x_max)
        print("y_min:",y_min)
        print("y_max:",y_max)
        print("Shape:",img_shape)

def check_dimensions(shapes,out_shape,keypoints):
    if(shapes[0] != out_shape or shapes[1] != out_shape):
        print("Error")
        print("width:",shapes[1])
        print("heigth:",shapes[0])
        print("\n")

    if(
        keypoints[0][0] > out_shape or keypoints[0][1] > out_shape or
        keypoints[1][0] > out_shape or keypoints[1][1] > out_shape or
        keypoints[2][0] > out_shape or keypoints[2][1] > out_shape or
        keypoints[3][0] > out_shape or keypoints[3][1] > out_shape
    ):
        print("keypoints:",keypoints)

