import cv2
import glob
import os
from skimage.color import rgba2rgb
from skimage.color import rgb2gray
from numpy import expand_dims
import tensorflow as tf
from tensorflow.keras.layers import Rescaling

def load_data(dir):
    
    # Retriving images path
    ImagesPath = glob.glob(dir + "*.jpg")
    images = []
    coords = []

    for filename in ImagesPath:
        # Read the image
        # In image we've array RGB that describe pixels
        image = cv2.imread(filename)

        # If image is RGBA we transform it in RGB
        if(image.shape[2] == 4):
            image = rgba2rgb(image)

        # conversion in grayscale
        converted = tf.image.rgb_to_grayscale(image)
        image = converted.numpy()
        scaler = Rescaling(scale=1.0 / 255)
        scaled_image = scaler(image)
        image = scaled_image.numpy()

        # we remove path and format from image name
        img_name = os.path.basename(filename)
        img_name = img_name[:-4]

        # txt_file is the file with coordinates
        txt_file = dir+img_name+".txt"

        # coordinates retrieving
        with open(txt_file) as f:
            lines = f.readlines()

        coord1=(float(lines[1].split(',')[5]),float(lines[1].split(',')[6]))
        coord2=(float(lines[2].split(',')[5]),float(lines[2].split(',')[6]))
        coord3=(float(lines[3].split(',')[5]),float(lines[3].split(',')[6]))
        coord4=(float(lines[4].split(',')[5]),float(lines[4].split(',')[6]))

        keypoints = [coord1,coord2,coord3,coord4]

        images.append(image)
        coords.append(keypoints)

    return images,coords