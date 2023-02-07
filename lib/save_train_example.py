from tensorflow.io import TFRecordWriter

class save_train_example:
    name_training_files = 0
    num_images = 0
    writer = None

    def __init__(self,num,dir):
        self.num_of_img_for_file = num
        self.dir_out = dir

    def save_example(self,serialized_example):
        if(self.num_images % self.num_of_img_for_file == 0):
            if(self.writer != None):
                self.writer.close()

            self.name_training_files += 1
            self.writer = TFRecordWriter(self.dir_out + str(self.name_training_files) + ".tfrecord")
    
        self.writer.write(serialized_example)

        # number of images added to files
        self.num_images += 1