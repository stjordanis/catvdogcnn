from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import mobilenet
from keras import callbacks
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
from os import listdir
from os import system as bash
from random import randint
from math import ceil
def val_cleanup(directories,tmp_dirs):
    bash("mv test/"+tmp_dirs[0]+"* "+directories[0])
    bash("mv test/"+tmp_dirs[1]+"* "+directories[1])
    bash("rm -r test")
def create_val(directories,tmp_dirs,num):
    imgs=[]
    
    dir_1_files=listdir(directories[0])
    dir_2_files=listdir(directories[1])
    
    files=dir_1_files
    bash("mkdir test/")
    bash("mkdir test/"+tmp_dirs[0])
    for i in range(0,num):
        file=files.pop(randint(0,len(files)-1))
        img = image.load_img(directories[0]+file, target_size=None)
        img = np.array(img)
        imgs.append(img/255)
        bash("mv "+directories[0]+file+" test/"+tmp_dirs[0])
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
    while True:
            imgs=[]
            if len(dir_1_files)<num or len(dir_2_files)<num:
                print("Hi!")
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
i=0
epochs=20
epochs=ceil(epochs/2)
dirs=["augmented/cats/","augmented/dogs/"]
tmp_dirs=["cats/","dogs/"]
x_test=np.array(create_val(dirs,tmp_dirs,32))
y_test=np.array(load_y(32))
Y = np.array(load_y(32))
for x in load_images(dirs,32):
    if i>=epochs:
        break
    X = np.array(x)
    model.fit(X, Y, epochs = 2, batch_size=32, validation_data=(x_test, y_test),callbacks=callbacks_list)
    i+=1
val_cleanup(dirs,tmp_dirs)
