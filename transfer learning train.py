from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import mobilenet
from keras import callbacks
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
from os import listdir

def load_images(directories=[]):
    imgs=[]
    for directory in directories:
        for file in listdir(directory):
            img = image.load_img(directory+file, target_size=None)
            img = np.array(img)
            imgs.append(img/255)
            del img
    return imgs
def load_y(directories):
    y_data=[]
    for i in range(0,len(listdir(directories[0]))):
        y_data.append([1,0])
    for i in range(0,len(listdir(directories[1]))):
        y_data.append([0,1])
    return y_data

img_width, img_height = 64, 64
shape = (img_width, img_height, 3)
mn=mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=shape)

dirs=["images/cats/","images/dogs/"]
Y = np.array(load_y(dirs))
X = np.array(load_images(dirs))
print(len(load_images(dirs)),"images")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

model = Sequential()
model.add(mn)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
print(model.summary())

savebest=callbacks.ModelCheckpoint(filepath='test.h5',monitor='val_loss',save_best_only=True)
callbacks_list=[savebest]
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs = 15, batch_size=64, validation_data=(x_test, y_test),callbacks=callbacks_list)
