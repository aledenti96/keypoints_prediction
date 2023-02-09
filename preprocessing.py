import os
import glob


# dir = "/media/alessandro/DATA/Universita/Magistrale/2_anno/LabIA/Dataset/DB3/"
dir = "/home/adenti/DB3/"

# change all ".JPG" in ".jpg"
def correct_format():
    filespath = glob.glob(dir+"*.JPG")

    for old_name in filespath:
        new_name = old_name[:-4] + ".jpg"
        os.rename(old_name,new_name)

# delete all .tif file (have points drawn)
def delete_tif():
    filespath = glob.glob(dir+"*.tif")
    
    for file in filespath:
        os.remove(file)

# delete file in which points aren't in right position
def delete_bad_files():

    filespath = glob.glob(dir+"*.txt")

    for file in filespath:

        with open(file) as f:
            lines = f.readlines()

        coord1=(float(lines[1].split(',')[5]),float(lines[1].split(',')[6]))
        coord2=(float(lines[2].split(',')[5]),float(lines[2].split(',')[6]))
        coord3=(float(lines[3].split(',')[5]),float(lines[3].split(',')[6]))
        coord4=(float(lines[4].split(',')[5]),float(lines[4].split(',')[6]))

        # checks if points are too near.
        # if they are probably the file is bad
        if (abs(coord1[0] - coord2[0]) < 50 and abs(coord1[1] - coord2[1]) < 50 and abs(coord2[0] - coord3[0]) < 50 and abs(coord2[1] - coord3[1]) < 50 and abs(coord2[0] - coord3[0]) < 50 and abs(coord2[1] - coord3[1]) < 50 and abs(coord3[0] - coord4[0]) < 50 and abs(coord3[1] - coord4[1]) < 50):

            os.remove(file)

            img_file = file[:-4] + ".jpg"
            os.remove(img_file)

# files are rename with a simpler name: es. 1.txt 1.jpg
def rename_files():

    count = 0
    filespath = glob.glob(dir+"*.txt")    

    for old_name_txt in filespath:

        old_name_image = old_name_txt[:-4] + ".jpg"

        # if does not exists a file jpg corrispondent with the txt, the ".txt" is not rename
        if(os.path.exists(old_name_image)):
            count += 1
            new_name = dir + str(count)

            os.rename(old_name_txt,new_name + ".txt")
            os.rename(old_name_image, new_name + ".jpg")
        else:
            print("File", old_name_txt, "has no match with a jpg")


# main
# functions are indipendet of each other. You can comment ones you don't need

correct_format()
delete_tif()
delete_bad_files()
rename_files()
