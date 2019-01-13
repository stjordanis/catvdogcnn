from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import mobilenet
from keras import callbacks
from keras.preprocessing import image
import numpy as np
from os import listdir
from os import system as bash
from random import randint
from math import ceil

def val_cleanup(directories,tmp_dirs):
    #move all the val images back to where they came from
    bash("mv test/"+tmp_dirs[0]+"* "+directories[0])
    bash("mv test/"+tmp_dirs[1]+"* "+directories[1])
    #delete the test folder
    bash("rm -r test")


def create_val(directories,tmp_dirs,num):
    imgs=[]
    
    #store a list of the files in each of the directories
    dir_1_files=listdir(directories[0])
    dir_2_files=listdir(directories[1])
    
    files=dir_1_files
    #make the required folders
    bash("mkdir test/")
    bash("mkdir test/"+tmp_dirs[0])
    #do num times
    for i in range(0,num):
        #remove a random file name from the list
        #removed to prevent image reuse
        file=files.pop(randint(0,len(files)-1))
        #load the file
        img = image.load_img(directories[0]+file, target_size=None)
        #convert it to a numpy array
        img = np.array(img)
        #/255 for data normalisiarion
        imgs.append(img/255)
        #moves the file to a temporary folder so it doesn't get used in training
        bash("mv "+directories[0]+file+" test/"+tmp_dirs[0])
        #deletes the img variable
        #probably not necesary
        del img
        
    files=dir_2_files
    bash("mkdir test/")
    bash("mkdir test/"+tmp_dirs[1])
    for i in range(0,num):
        file=files.pop(randint(0,len(files)-1))
        img = image.load_img(directories[1]+file, target_size=None)
        img = np.array(img)
        imgs.append(img/255)
        bash("mv "+directories[1]+file+" test/"+tmp_dirs[1])
        del img

    return imgs   
    
def load_images(directories,num):
    dir_1_files=listdir(directories[0])
    dir_2_files=listdir(directories[1])
    #so we can get data forever
    while True:
            imgs=[]
            #"refills" the files lists to allow for the reuse of images if there isn't enough images in either of the file lists
            if len(dir_1_files)<num or len(dir_2_files)<num:
                dir_1_files=listdir(directories[0])
                dir_2_files=listdir(directories[1])

            files=dir_1_files
            for i in range(0,num):
                file=files.pop(randint(0,len(files)-1))
                img = image.load_img(directories[0]+file, target_size=None)
                img = np.array(img)
                imgs.append(img/255)
                del img
            files=dir_2_files
            for i in range(0,num):
                file=files.pop(randint(0,len(files)-1))
                img = image.load_img(directories[1]+file, target_size=None)
                img = np.array(img)
                imgs.append(img/255)
                del img
            #yields the images folder
            yield imgs
def load_y(num):
    y_data=[]
    for i in range(0,num):
        y_data.append([1,0])
    for i in range(0,num):
        y_data.append([0,1])
    return y_data


img_width, img_height = 64, 64
shape = (img_width, img_height, 3)
#load MobileNet pretrained on imagenet with input_shape of the shape variable
mn=mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=shape)

model = Sequential()
model.add(mn)
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
#tells the user the model summary for sanity checking
print(model.summary())

#same filename each time to force keras to overwrite the file each time and minimise the amount of disk space used
savebest=callbacks.ModelCheckpoint(filepath='generator_model.h5',monitor='val_loss',save_best_only=True)
callbacks_list=[savebest]
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


i=0
#set the number of epochs you want here
epochs=20
#as we do two epochs per "batch" of data halve it and round decimals up as there is little harm in doing an extra epoch
epochs=ceil(epochs/2)

#change your directories here if necesary
dirs=["augmented/cats/","augmented/dogs/"]
tmp_dirs=["cats/","dogs/"]

#creates val data
x_test=np.array(create_val(dirs,tmp_dirs,32))
y_test=np.array(load_y(32))

#creates train Y data
Y = np.array(load_y(32))

#this amounts to a forever loop due to the while true in load_images
#loads our train X data
for x in load_images(dirs,32):
    #ends the loop if we have done enough epochs
    if i>=epochs:
        break

    #converts X to a numpy array
    X = np.array(x)

    #trains for 2 epochs
    model.fit(X, Y, epochs = 2, batch_size=32, validation_data=(x_test, y_test),callbacks=callbacks_list)
    i+=1

#at the end of training, cleanup
val_cleanup(dirs,tmp_dirs)
