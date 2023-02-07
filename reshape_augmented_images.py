import os
import glob
from tensorflow.data import TFRecordDataset
from tensorflow.io import TFRecordWriter

from lib import reshape
from lib import deserialize_example_reshape as des
from lib import serialize_example as ser
from lib import save_train_example as save_tr


# Objective is reshaping keeping apart testing, validation and training examples

cluster = False
# Keep in mind that you starts from a 1000x1000 image and that factor scaling have to be an integer.
# From 1000x1000 to 500x500 factor scaling will be 2. So... that's ok
resize_shape = 500

if(cluster):
    num_of_img_for_file = 140
    dir_in = "/home/adenti/data_augmentation_tfrecord/datafile/"
    dir_out = "/home/adenti/reshape/sized_images/"+str(resize_shape)+"/"
    dir_out_training = dir_out + "training/"
    dir_out_testing = dir_out + "testing/"
    dir_out_validation = dir_out + "validation/"
else:
    num_of_img_for_file = 2
    dir_in = "/home/alessandro/Scrivania/dataset/"
    dir_out = "/home/alessandro/Scrivania/sized_images/"+str(resize_shape)+"/"
    dir_out_training = dir_out + "training/"
    dir_out_testing = dir_out + "testing/"
    dir_out_validation = dir_out + "validation/"



if(os.path.exists(dir_in) == False):
    print("\n")
    print("Error: Directory not exists")
    exit()

# image will be (out_shape,out_shape)
# keep this to 1000. If you need a smaller image, change resize_shape
out_shape = 1000

if(resize_shape > 0):
    final_resize_factor = out_shape/resize_shape

    if((final_resize_factor - int(final_resize_factor)) != 0):
        print("Final resize factor must be an integer")
        exit()
    final_resize_factor = int(final_resize_factor)
    print("Final_resize_factor: ",final_resize_factor)
else:
    final_resize_factor = 0

training_filenames = glob.glob(dir_in + "training/" + "*.tfrecord")
test_filename = dir_in + "testing/test_set.tfrecord"
validation_filename = dir_in + "validation/validation_set.tfrecord"


# We first lasts delete created file
files = glob.glob(dir_out_training+"*.tfrecord")
for f in files:
    os.remove(f)

files = glob.glob(dir_out_testing + "*.tfrecord")
for f in files:
        os.remove(f)

files = glob.glob(dir_out_validation + "*.tfrecord")
for f in files:
        os.remove(f)

#TRAINING 
# retrieving the dataset from file TFRecord
ds_bytes = TFRecordDataset(training_filenames)
# deserialization of images
dataset = ds_bytes.map(des.deserialize_example)

count_ok = 0
count_not_ok = 0
save = save_tr.save_train_example(num_of_img_for_file,dir_out_training)
for image,keypoints in dataset.as_numpy_iterator():
    res_image,res_keypoints = reshape.reshape(image,keypoints,out_shape,final_resize_factor)
    if(len(res_image) == 0):
        print("Image is not resizable")
        count_not_ok +=1
    else:
        serialized_example = ser.serialize_example(res_image,res_keypoints)
        save.save_example(serialized_example)
        count_ok += 1

print("Number of usable training images:",count_ok)
print("Number of not usable training images:",count_not_ok)

#TEST
# retrieving the dataset from file TFRecord
ds_bytes = TFRecordDataset(test_filename)
# deserialization of images
dataset = ds_bytes.map(des.deserialize_example)

count_ok = 0
count_not_ok = 0
writer = TFRecordWriter(dir_out_testing + "test_set.tfrecord")
for image,keypoints in dataset.as_numpy_iterator():
    res_image,res_keypoints = reshape.reshape(image,keypoints,out_shape,final_resize_factor)
    if(len(res_image) == 0):
        print("Image is not resizable")
        count_not_ok +=1
    else:
        serialized_example = ser.serialize_example(res_image,res_keypoints)
        writer.write(serialized_example)
        count_ok += 1

print("Number of usable testing images:",count_ok)
print("Number of not usable testing images:",count_not_ok)


#VALIDATION
# retrieving the dataset from file TFRecord
ds_bytes = TFRecordDataset(test_filename)
# deserialization of images
dataset = ds_bytes.map(des.deserialize_example)

count_ok = 0
count_not_ok = 0
writer = TFRecordWriter(dir_out_validation + "validation_set.tfrecord")
for image,keypoints in dataset.as_numpy_iterator():
    res_image,res_keypoints = reshape.reshape(image,keypoints,out_shape,final_resize_factor)
    if(len(res_image) == 0):
        print("Image is not resizable")
        count_not_ok +=1
    else:
        serialized_example = ser.serialize_example(res_image,res_keypoints)
        writer.write(serialized_example)
        count_ok += 1

print("Number of usable validation images:",count_ok)
print("Number of not usable validation images:",count_not_ok)