from random import randint as rand
import albumentations as A
import math

def area_delimitation(image,keypoints):
    # This values indicate min area that we can't crop
    x_min = min(keypoints[0][0],keypoints[1][0],keypoints[2][0],keypoints[3][0])
    x_max = max(keypoints[0][0],keypoints[1][0],keypoints[2][0],keypoints[3][0])
    y_min = min(keypoints[0][1],keypoints[1][1],keypoints[2][1],keypoints[3][1])
    y_max = max(keypoints[0][1],keypoints[1][1],keypoints[2][1],keypoints[3][1])


    # Now that we know the min necessary area, we can randomize for the possible region

    if(x_min > 0): # if x_min is equal to 0, we can't crop less than the existent border
        rounded = math.floor(x_min) 
        if(rounded < x_min): # if x_min is a float, its floor is smaller than x_min self...
            cut_x_min = rand(0,rounded) # ...we can keep that bound as max
        else:
            cut_x_min = rand(0,(x_min-1)) # if x_min is an integer, its safer put bound at x_min -1. In this way we'are sure the keypoints isn't cropped.
    else:
        cut_x_min = 0

    # the same reasoning is made for other three points.

    if(x_max < image.shape[1]):
        rounded = math.ceil(x_max)
        if(rounded > x_max):
            cut_x_max = rand(rounded,image.shape[1])
        else:
            cut_x_max = rand((x_max + 1),image.shape[1])
    else:
        cut_x_max = image.shape[1]

    if(y_min > 0):
        rounded = math.floor(y_min)
        if(rounded < y_min):
            cut_y_min = rand(0,rounded)
        else:
            cut_y_min = rand(0,(y_min-1))
    else:
        cut_y_min = 0

    if(y_max < image.shape[0]):
        rounded = math.ceil(y_max)
        if(rounded > y_max):
            cut_y_max = rand(rounded,image.shape[0])
        else:
            cut_y_max = rand((y_max + 1),image.shape[0])
    else:
        cut_y_max = image.shape[0]

    # return x_min,x_max,y_min,y_max
    return cut_x_min,cut_x_max,cut_y_min,cut_y_max


def crop_image(image,keypoints):

    # bounding areas (not croppable)
    left_x,right_x,down_y,up_y = area_delimitation(image,keypoints)

    pipeline_crop = A.Compose(
        [A.Crop(x_min = left_x, y_min = down_y,x_max = right_x,y_max = up_y)],
        keypoint_params=A.KeypointParams(format='xy')
    )

    image_cropped = pipeline_crop(image = image,keypoints = keypoints)

    # controls to evitate an output keypoints number lower than 4
    before_shape = image.shape
    after_shape = image_cropped['image'].shape

    if(len(image_cropped['keypoints']) < 4):
        print("Before shape:", before_shape)
        print("After shape:", after_shape)
        print("Before keypoints:",keypoints)
        print("After keypoints:", image_cropped['keypoints'])
        print("\n")
        print("left_x:",left_x)
        print("right_x:",right_x)
        print("down_y:",down_y)
        print("up_y:",up_y)
        print("\n\n")

    return image_cropped