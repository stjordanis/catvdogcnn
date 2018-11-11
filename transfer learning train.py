from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.applications import mobilenet
import numpy as np
from os import listdir
from keras.preprocessing import image

def load_images(directories=[]):
    imgs=[]
    for directory in directories:
        for file in listdir(directory):
            img = image.load_img(directory+file, target_size=None)
            img = np.array(img)
            imgs.append(img/255)
            del img
    return imgs

y_data=[]
for i in range(0,132):
    y_data.append([1,0])
for i in range(0,132):
    y_data.append([0,1])

img_width, img_height = 64, 64
mn=mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
x_data=["images/cats/","images/dogs/"]
Y = np.array(y_data)
X = np.array(load_images(x_data))
input_shape = (img_width, img_height, 3)
print(X.shape)
print(len(load_images(x_data)))
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
model = Sequential()
model.add(mn)
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs = 15, batch_size=64, validation_data=(x_test, y_test))
model.save('model.h5')
