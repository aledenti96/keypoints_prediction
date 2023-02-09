from tensorflow.keras.models import save_model, load_model
import glob
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Rescaling
import numpy as np

from lib import load_data as ld
from lib import reshape

KEYPOINT_COLOR = (0,255,0)

def vis_keypoints(image,keypoints,resize_shape,color=KEYPOINT_COLOR,diameter=15):
    img = image.copy()

    reshaped_keypoints = keypoints.reshape(4,2)

    print("Image[i]:",image.shape)
    image = img.reshape(resize_shape,resize_shape)

    # bring points to original dimension
    # if(resize_shape != False):
        # scaler = Rescaling(scale = resize_shape)
        # keypoints = scaler(reshaped_keypoints)
    keypoints = [x*resize_shape for x in reshaped_keypoints] 
    # print("Keypoints:",keypoints)

    for (x,y) in keypoints:
        cv2.circle(image, (int(x),int(y)), diameter, color, -1)

    plt.imshow(image)
    plt.show()

dir_in = "/home/alessandro/Scrivania/prediction/"
# dir_model = "/media/alessandro/DATA/Universita/Magistrale/2_anno/LabIA/results/7_relu_last_layer/"
dir_model = "/media/alessandro/DATA/Universita/Magistrale/2_anno/LabIA/results/500_pixel_scalati/batteria_1/3/"

out_shape = 1000

# Keep in mind that you starts from a 1000x1000 image and that factor scaling have to be an integer.
# From 1000x1000 to 500x500 factor scaling will be 2. So... that's ok
resize_shape = 500

if(resize_shape > 0):
    final_resize_factor = out_shape/resize_shape

    if((final_resize_factor - int(final_resize_factor)) != 0):
        print("Final resize factor must be an integer")
        exit()
    final_resize_factor = int(final_resize_factor)
    print("Final_resize_factor: ",final_resize_factor)
else:
    final_resize_factor = 0

images,keypoints = ld.load_data(dir_in)
# print("Image:",images[0])

original_examples = []
for i in range(len(images)):
    res_image,res_keypoints = reshape.reshape(images[i],keypoints[i],out_shape,final_resize_factor)
    if(len(res_image)==0):
        print("image not resizable")
    else:
        # print("Resize img:",res_image)
        original_examples.append((res_image,res_keypoints))


model = load_model(dir_model + "model_early_stopping_shuffle_low_lr.h5")

#print("Images:",images)

for example in original_examples:
    plt.title("Original points")
    vis_keypoints(example[0],example[1],resize_shape)
    
    image_batch = np.expand_dims(example[0],0)
    predicted_keypoints = model.predict(image_batch,batch_size=1,verbose=0)

    print("predicted keypoints:",predicted_keypoints)

    plt.title("Predicted points")
    vis_keypoints(example[0],predicted_keypoints,resize_shape)

    print("\n")




