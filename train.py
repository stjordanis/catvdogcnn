from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import mobilenet
from keras import callbacks
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
from os import listdir

def load_images(directories=[]):
    #create a blank list to store the images in
    imgs=[]
    #loop over the list of directories we have been passed
    for directory in directories:
        #loop over the files that are contained in the directory
        for file in listdir(directory):
            #load the image in the directory directory with the filename file
            img = image.load_img(directory+file, target_size=None)
            #convert it to a numpy array
            img = np.array(img)
            #/255 for data normalization
            #append to the imgs list
            imgs.append(img/255)
            #delete the img variable to save RAM
            del img
    return imgs


def load_y(directories):
    y_data=[]
    #add [1,0] (0 one hot encoded) to y_data the same amount of times as the number of files in the first directory in direcotories
    for i in range(0,len(listdir(directories[0]))):
        y_data.append([1,0])
    #same but with second direcotry
    for i in range(0,len(listdir(directories[1]))):
        y_data.append([0,1])
    return y_data

img_width, img_height = 64, 64
shape = (img_width, img_height, 3)
#load MobileNet pretrained on imagenet with input_shape of the shape variable
mn=mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=shape)

dirs=["images/cats/","images/dogs/"]
#creates the Y data
Y = np.array(load_y(dirs))
#loads all the images
X = np.array(load_images(dirs))
#tells the user the total number of images
print(len(load_images(dirs)),"images")

#creates train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

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

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\
model.fit(X, Y, epochs = 15, batch_size=64, validation_data=(x_test, y_test),callbacks=callbacks_list)
