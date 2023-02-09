from tensorflow.data import TFRecordDataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.callbacks import EarlyStopping
import glob
import math
import matplotlib.pyplot as plt
import os

import deserialize_example as des

cluster = True
shape_image = 500
input_shape = (shape_image,shape_image,1)
num_classes = 8
files_out_names = "early_stopping_shuffle_low_low_lr"

if(cluster):
    # directory in my home in cluster
    # src = "/home/adenti/reshape/sized_images/"+str(shape_image)+"/"
    # # that's directory of ssd memory of the node
    # dest = '/tmp'
    # name_dir = "/500/"
    dir_in = "/home/adenti/reshape/sized_images/"+str(shape_image)+"/"
    num_epochs = 70
    model_path = "/home/adenti/"
    batch_size_training = 30

    # # if not exists a directory 500, copy her
    # if(not os.path.exists(dest + name_dir)):
    #     # os.system("rm -r " + dest + name_dir)

    #     # show content of /tmp
    #     # os.system("ls /tmp")

    #     # copy in the content of "dir_in"
    #     os.system("cp -r " + src + " " + dest + name_dir)

    #     # show content of tmp
    #     os.system("ls /tmp")

    # dir_in = dest + name_dir
else:
    dir_in = "/home/alessandro/Scrivania/sized_images/"+str(shape_image)+"/"
    num_epochs = 5
    batch_size_training = 10
    model_path = "/home/alessandro/Scrivania/"

# learning rate parameters
# initial_learning_rate = 0.96
initial_learning_rate = 0.9
# decay_rate = 0.01
decay_step = num_epochs*batch_size_training
end_learning_rate = 0.005

dir_in_training = dir_in + "training/"
dir_in_validation = dir_in + "validation/"
dir_in_testing = dir_in + "testing/"

# loading training dataset
filenames_train = glob.glob(dir_in_training + "*.tfrecord")
ds_bytes = TFRecordDataset(filenames_train)
dataset_training = ds_bytes.map(des.deserialize_example)
dataset_training = dataset_training.shuffle(5340, reshuffle_each_iteration=True)

# loading validating dataset
filename_validate = glob.glob(dir_in_validation + "*.tfrecord")
ds_bytes = TFRecordDataset(filename_validate)
dataset_validation = ds_bytes.map(des.deserialize_example)

# batching data
dataset_training = dataset_training.batch(batch_size_training)
dataset_validation = dataset_validation.batch(1)

# Model definition

model = Sequential()

# block 1
model.add(Conv2D(32, (3,3), strides = (1,1), padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3,3), strides = (1,1), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# block 2
model.add(Conv2D(64, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# block 3
model.add(Conv2D(128, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

#block 4
model.add(Conv2D(128, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# block 5
model.add(Conv2D(256, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# block 6
model.add(Conv2D(256, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# block 7
model.add(Conv2D(512, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

#block 8
model.add(Conv2D(512, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), strides = (1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# 1 dense layer
model.add(Flatten())
model.add(Dense(128,activation='relu'))

# finale dense layer
model.add(Dense(num_classes))

model.summary()


# Remember: is computed as
# lr = initial_learning_rate * decay_rate ^ (epoch / decay_step)

# lr = schedules.ExponentialDecay(initial_learning_rate,decay_step,decay_rate,staircase=False)
lr = schedules.PolynomialDecay(
    initial_learning_rate,
    decay_step,
    end_learning_rate
)

callback = EarlyStopping(monitor='val_loss',mode="min",patience=5,restore_best_weights=True)
optimizer = Adam(learning_rate=lr)

# Compiling model
#model.compile(loss='mse',optimizer=Adam(learning_rate = initial_learning_rate))
model.compile(loss='mse',optimizer=optimizer)

# Fit model
print("Fit on Train Dataset")
history = model.fit(dataset_training, epochs=num_epochs, batch_size=batch_size_training,callbacks=[callback],validation_data = dataset_validation,use_multiprocessing=True)

model.save(model_path + "model_" + files_out_names + ".h5",save_format='h5')

# plot graph
x = []
y = []
for i in range(len(history.history['loss'])):
    x.append(i)
    y.append(history.history['loss'][i])

plt.title("Training loss")
plt.plot(x,y)
plt.axis([0,num_epochs,0,2])

if(cluster):
    plt.savefig("Training_loss_" + files_out_names + ".png")
else:
    plt.show()

x = []
y = []
for i in range(len(history.history['val_loss'])):
    x.append(i)
    y.append(history.history['val_loss'][i])

plt.title("Validation loss")
plt.plot(x,y)
plt.axis([0,num_epochs,0,2])

print("x: ",x)
print("y: ",y)

if(cluster):
    plt.savefig("Validation_loss_" + files_out_names + ".png")
else:
    plt.show()


# loading test dataset
filename_test = glob.glob(dir_in_testing + "*.tfrecord")
ds_bytes = TFRecordDataset(filename_test)
dataset_test = ds_bytes.map(des.deserialize_example)

# batch dataset
dataset_test = dataset_test.batch(1)

# Evaluate model on test dataset
print("Evaluate on Test Dataset")
eval_loss = model.evaluate(dataset_test)

print("Evaluation loss:", eval_loss)
