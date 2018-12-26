from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import mobilenet
from keras import callbacks
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
from os import listdir
from random import randint

def load_images(directories,num):
    #create a blank list to store the images in
    dir_1_files=listdir(directories[0])
    dir_2_files=listdir(directories[1])
    #loop over the list of directories we have been passed
    while True:
            imgs=[]
            if len(dir_1_files)<num or len(dir_2_files)<num:
                print("Hi!")
                dir_1_files=listdir(directories[0])
                dir_2_files=listdir(directories[1])
            files=dir_1_files
            #loop over the files that are contained in the directory
            for i in range(0,num):
                file=files.pop(randint(0,len(files)-1))
                #load the image in the directory directory with the filename file
                img = image.load_img(directories[0]+file, target_size=None)
                #convert it to a numpy array
                img = np.array(img)
                #/255 for data normalization
                #append to the imgs list
                imgs.append(img/255)
                #delete the img variable to save RAM
                del img
            files=dir_2_files
            #loop over the files that are contained in the directory
            for i in range(0,num):
                file=files.pop(randint(0,len(files)-1))
                #load the image in the directory directory with the filename file
                img = image.load_img(directories[1]+file, target_size=None)
                #convert it to a numpy array
                img = np.array(img)
                #/255 for data normalization
                #append to the imgs list
                imgs.append(img/255)
                #delete the img variable to save RAM
                del img
            yield imgs
def load_y(directories,num):
    y_data=[]
    #add [1,0] (0 one hot encoded) to y_data the same amount of times as the number of files in the first directory in direcotories
    for i in range(0,num):
        y_data.append([1,0])
    #same but with second direcotry
    for i in range(0,num):
        y_data.append([0,1])
    return y_data

img_width, img_height = 64, 64
shape = (img_width, img_height, 3)
#load MobileNet pretrained on imagenet with input_shape of the shape variable
mn=mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=shape)
#adds dense layers after mobilenet
model = Sequential()
model.add(mn)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
#tells the user the model summary for sanity checking
print(model.summary())

#same filename each time to force keras to overwrite the file each time and minimise the amount of disk space used
savebest=callbacks.ModelCheckpoint(filepath='model.h5',monitor='val_loss',save_best_only=True)
callbacks_list=[savebest]
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#loads images
dirs=["augmented/cats/","augmented/dogs/"]
Y = np.array(load_y(dirs,32))
for x in load_images(dirs,32):
    X = np.array(x)
    #creates train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    model.fit(X, Y, epochs = 2, batch_size=32, validation_data=(x_test, y_test),callbacks=callbacks_list)
