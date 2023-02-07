import albumentations as A
from matplotlib import pyplot as plt
import cv2
from random import randint as rand
from sklearn.model_selection import train_test_split
import numpy as np
import glob
import os
import tensorflow as tf
from tensorflow.io import TFRecordWriter

from lib import load_data as ld
from lib import crop_helpers as ch
from lib import serialize_example as ser
from lib import save_train_example as save_tr


def keypoints_check(before_key,after_key,func,before_shape,after_shape):
    if(len(after_key) < 4):
        print(func + " gives out " + str(len(after_key)) + " keypoints")
        print("Before shape:", before_shape)
        print("After shape:", after_shape)
        print("Before keypoints:",before_key)
        print("After keypoints:", after_key)
    

# parameters
cluster = False

if(cluster):
    dir = "/home/adenti/DB3/"
    dir_out = "/home/adenti/data_augmentation_tfrecord/datafile/"
    num_of_img_for_file = 140 # number of examples per file
else:
    dir = "/home/alessandro/Scrivania/albumentation_test/"
    dir_out = "/home/alessandro/Scrivania/dataset/"
    num_of_img_for_file = 5 # number of examples per file

name_test_file = "test_set"
name_validation_file = "validation_set"
test_data_perc = 0.2 # percentage of test data
validation_data_perc = 0.2 # percentage of validation set

if(os.path.exists(dir) == False):
    print("\n")
    print("Error: Directory not exists")
    exit()

# variables
name_training_files = 0
num_images = 0

# to save tfrecord files we first delete lasts created files
files = glob.glob(dir_out + "training/" +"*.tfrecord")
for f in files:
    os.remove(f)

files = glob.glob(dir_out + "testing/" +"*.tfrecord")
for f in files:
    os.remove(f)

files = glob.glob(dir_out + "validation/" +"*.tfrecord")
for f in files:
    os.remove(f)

# Load images and coordinates from dir
images,coords = ld.load_data(dir)

# Splitting data into training, test and validation
x = np.array(images)
y = np.array(coords)
# print("images:",x)
# print("coords:",y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_data_perc)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=validation_data_perc) # use same functions to get also validation set

# pipeline for horizontal flip
horiz_flip = A.Compose(
    [A.HorizontalFlip(p=1)],
    keypoint_params=A.KeypointParams(format='xy')
)

# pipeline for vertical flip
vertic_flip = A.Compose(
    [A.VerticalFlip(p=1)],
    keypoint_params=A.KeypointParams(format='xy')
)

# pipeline for rotation
rotation = A.Compose(
    [A.Rotate(limit = 30)],
    keypoint_params=A.KeypointParams(format='xy')
)

# Creation of test data tfrecord file

writer = TFRecordWriter(dir_out + "testing/" + name_test_file + ".tfrecord")

# For each element of test example, serialized it and save it to the file
for i in range(len(x_test)):
    serialized_example = ser.serialize_example(x_test[i],y_test[i])
    writer.write(serialized_example)

writer.close()

# Creation of validation set tfrecord file

writer = TFRecordWriter(dir_out + "validation/" + name_validation_file + ".tfrecord")

# For each element of validation example, serialized it and save it to the file
for i in range(len(x_validation)):
    serialized_example = ser.serialize_example(x_validation[i],y_validation[i])
    writer.write(serialized_example)

writer.close()

# Creation of training data tfrecord file

save = save_tr.save_train_example(num_of_img_for_file,(dir_out + "training/"))

for i in range(len(x_train)):
    image = x_train[i]
    keypoints = y_train[i]

    original_shape = image.shape

    keypoints = [(round(k[0],0),round(k[1],0)) for k in keypoints]

    # saving serialized image to the file
    serialized_example = ser.serialize_example(image,keypoints)
    save.save_example(serialized_example)
    
    # rotation pipeline application and saving rotated image
    image_rotation = rotation(image=image,keypoints=keypoints)
    if(len(image_rotation['keypoints']) < 4):
        print("Rotation correction")
        image_rotation['image'] = image
        image_rotation['keypoints'] = keypoints
    # print("Image shape:",image.shape)
    serialized_example = ser.serialize_example(image_rotation['image'],image_rotation['keypoints'])
    save.save_example(serialized_example)

    # horizontal flip pipeline application and saving horizontally flipped image
    horizontal_flip = horiz_flip(image=image,keypoints=keypoints)
    # print("Image shape:",horizontal_flip['image'].shape)
    # new_shape = horizontal_flip['image'].shape
    # keypoints_check(keypoints,horizontal_flip['keypoints'],"horizontal flip",original_shape,new_shape)
    serialized_example = ser.serialize_example(horizontal_flip['image'],horizontal_flip['keypoints'])
    save.save_example(serialized_example)

    # vertical pipeline application and saving vertically flip image
    vertical_flip = vertic_flip(image=image,keypoints=keypoints)
    # print("Image shape:",vertical_flip['image'].shape)
    # new_shape = vertical_flip['image'].shape
    # keypoints_check(keypoints,vertical_flip['keypoints'],"vertical flip",original_shape,new_shape)
    serialized_example = ser.serialize_example(vertical_flip['image'],vertical_flip['keypoints'])
    save.save_example(serialized_example)

    # pipeline for crop
    # (cropping image must not to cut off keypoints so this pipeline is customize for every single image)

    # find area boundaries so as not exclude keypoints
    x_min,x_max,y_min,y_max = ch.area_delimitation(image,keypoints)

    # crop
    rand_crop = A.Compose(
        [A.Crop(x_min,y_min,x_max,y_max)],
        keypoint_params=A.KeypointParams(format='xy')
    )

    random_crop = rand_crop(image=image,keypoints=keypoints)
    # print("Image shape:",random_crop['image'].shape)
    # new_shape = random_crop['image'].shape
    # keypoints_check(keypoints,random_crop['keypoints'],"random crop",original_shape,new_shape)
    serialized_example = ser.serialize_example(random_crop['image'],random_crop['keypoints'])
    save.save_example(serialized_example)


    # MIXED PIPELINE

    # horizontal flip e crop

    # flipped img
    horizontal_flipped_cropped = ch.crop_image(horizontal_flip['image'],horizontal_flip['keypoints'])
    # print("Image shape:",horizontal_flipped_cropped['image'].shape)
    # original_shape = horizontal_flip['image'].shape
    # new_shape = horizontal_flipped_cropped['image'].shape
    # keypoints_check(random_crop['keypoints'],horizontal_flipped_cropped['keypoints'],"horizontal flipped cropped",original_shape,new_shape)
    serialized_example = ser.serialize_example(horizontal_flipped_cropped['image'],horizontal_flipped_cropped['keypoints'])
    save.save_example(serialized_example)

    # print("KeyP after horizontal crop:",horizontal_flipped_cropped['keypoints'])

    # vertical flip and crop
    vertical_flipped_cropped = ch.crop_image(vertical_flip['image'],vertical_flip['keypoints'])
    # print("Image shape:",vertical_flipped_cropped['image'].shape)
    # original_shape = vertical_flip['image'].shape
    # new_shape = vertical_flipped_cropped['image'].shape
    # keypoints_check(random_crop['keypoints'],vertical_flipped_cropped['keypoints'],"vertical flipped cropped",original_shape,new_shape)
    serialized_example = ser.serialize_example(vertical_flipped_cropped['image'],vertical_flipped_cropped['keypoints'])
    save.save_example(serialized_example)

    # rotated and crop
    rotated_cropped = ch.crop_image(image_rotation['image'],image_rotation['keypoints'])
    # print("Image shape:",rotated_cropped['image'].shape)
    # original_shape = image_rotation['image'].shape
    # new_shape = rotated_cropped['image'].shape
    # keypoints_check(random_crop['keypoints'],rotated_cropped['keypoints'],"rotated cropped",original_shape,new_shape)
    serialized_example = ser.serialize_example(rotated_cropped['image'],rotated_cropped['keypoints'])
    save.save_example(serialized_example)

writer.close()    


