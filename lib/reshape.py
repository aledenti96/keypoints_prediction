from tensorflow.data import TFRecordDataset
from tensorflow.io import TFRecordWriter
import tensorflow as tf
# import glob
import cv2
from tensorflow.io import TFRecordWriter
# import os
import math
from tensorflow.keras.layers import Rescaling
import numpy as np

from lib import resize

def reshape(image,keypoints,out_shape,final_resize_factor):
    # Resizing image as near as possible to out_shape.
    # Image remain greater than (out_shape,out_shape)
    rsz_image, rsz_keypoints = resize.resize(image,keypoints,out_shape)

    # Find keypoints near the border of image
    bottom = min(rsz_keypoints[0][1],rsz_keypoints[1][1],rsz_keypoints[2][1],rsz_keypoints[3][1])
    top = max(rsz_keypoints[0][1],rsz_keypoints[1][1],rsz_keypoints[2][1],rsz_keypoints[3][1])
    left = min(rsz_keypoints[0][0],rsz_keypoints[1][0],rsz_keypoints[2][0],rsz_keypoints[3][0])
    right = max(rsz_keypoints[0][0],rsz_keypoints[1][0],rsz_keypoints[2][0],rsz_keypoints[3][0])

    # Keypoints are in the form (x,y), so x is the width and y is height.
    # To remain coherent is good thing invert image_shape coordinates (that is in the form (y,x))
    # img_shape = (math.floor(rsz_image.shape[1]),math.floor(rsz_image.shape[0]))
    img_shape = ((rsz_image.shape[1]),(rsz_image.shape[0]))

    # Control keypoints isn't on left border (if it is, left crop will set to 0)
    if(left > 0):
        rounded = math.floor(left)
        # if the floor of "left keypoints" is minor respect "left", means that this value is float,
        # so to make safer computing of "non-croppable-area", we keep the floor of the value
        if(left > rounded):
            left = rounded
        else:
        # if the value it's an integer, to make safer the computing of non-croppable-area, we decrement left value by 1 
            left -= 1
    else:
        left = 0

    # this means "stay to the side of the corn" :)

    # the same is made for right,bottom and top

    if(right < img_shape[0]):
        rounded = math.ceil(right)
        if(right < rounded):
            right = rounded
        else:
            right += 1
    else:
        right = img_shape[0]

    if(bottom > 0):
        rounded = math.floor(bottom)
        if(bottom > rounded):
            bottom = rounded
        else:
            bottom -= 1
    else:
        bottom = 0

    if(top < img_shape[1]):
        rounded = math.ceil(top)
        if(top < rounded):
            top = rounded
        else:
            top += 1
    else:
        top = img_shape[1]

    # if the following condition is respected, image can be cropped in (out_shape,out_shape)
    if((right - left) < out_shape and (top - bottom) < out_shape):

        # keep count of images "croppable"
        # count_ok += 1

        # find random boundaries (but not so random) to bring image dimension to be exactly (out_shape,out_shape)
        x_max,x_min,y_max,y_min = resize.random_boundaries(left,right,top,bottom,img_shape,out_shape)
        # check if boundaries respects rules
        resize.check_boundaries(x_min,x_max,y_min,y_max,img_shape)

        # print("Resize Image:",rsz_image.shape)

        # crop image
        cropped_image,cropped_keypoints = resize.crop(x_min,x_max,y_min,y_max,rsz_image,rsz_keypoints)

        # check if image respect (out_shape,out_shape)
        resize.check_dimensions(cropped_image.shape,out_shape,cropped_keypoints)


        if(final_resize_factor > 0):
            width = int(cropped_image.shape[1] / final_resize_factor)
            height = int(cropped_image.shape[1] / final_resize_factor)
            dim = (width,height)
            # print("Dim:",dim)

            final_resized_image = cv2.resize(cropped_image, dim, interpolation = cv2.INTER_AREA)
            final_resized_image = np.expand_dims(final_resized_image,2)
            # print("Image expanded:",image)
            # print("Image expanded reshaped:",final_resized_image.shape)
            final_resized_keypoints = [(i[0]/final_resize_factor,i[1]/final_resize_factor) for i in cropped_keypoints]

            # check if image respect (out_shape,out_shape)
            resize.check_dimensions(final_resized_image.shape,width,final_resized_keypoints)

            cropped_image = final_resized_image
            cropped_keypoints = final_resized_keypoints            

        # reshaping keypoints in way they're all in a lists
        np_cropped_keypoints = np.array(cropped_keypoints)
        reshaped_keypoints = np_cropped_keypoints.reshape(1,1,2*4)
        #print("Reshaped_keypoints:",reshaped_keypoints)

        if(final_resize_factor > 0):
            # declaration of a scaler to bring all points have value between 0 and 1
            scaler = Rescaling(scale = 1.0 / width)
        else:
            scaler = Rescaling(scale = 1.0 / out_shape)

        # scaling keypoints
        scaled_keypoints = scaler(reshaped_keypoints)

        return cropped_image,scaled_keypoints.numpy()
    else:
            # keep count of image not "croppable"
        return [],[]
        # print("\n")


