#code modified from https://medium.com/@kylepob61392/airplane-image-classification-using-a-keras-cnn-22be506fdb53
#this code is a mess so I'm not going to even bother to try to fully comment it

from keras.models import load_model
import numpy as np
from os import listdir
from keras.preprocessing import image
from math import sqrt, ceil
import matplotlib.pyplot as plt

#same image loading in train.py
def load_images(directories=[]):
    imgs=[]
    for directory in directories:
        for file in listdir(directory):
            img = image.load_img(directory+file, target_size=None)
            img = np.array(img)
            imgs.append(img/255)
            del img
    return imgs

#set to the number of images you have in each folder
num=132
y_data=[]
for i in range(0,num):
    y_data.append([0])
for i in range(0,num):
    y_data.append([1])

#loads the model and x and y data
model=load_model("model.h5")
x_data=["images/cats/","images/dogs/"]
Y = np.array(y_data)
X = np.array(load_images(x_data))

#make predictions
test_predictions=[]
predictions = model.predict(X)
for el in predictions:
    test_predictions.append(el.argmax())

#don't even ask why the bulk of the code is in a subroutine but it's too much effort to fix at the moment    
def visualize_incorrect_labels(x_data, y_real, y_predicted):
    figure = plt.figure()
    #initalise the descriptively named variables
    i=0
    j=1
    p=0

    #calucates how many images we will be showing
    for el in x_data:
        #if the ith predction was correct do nothing
        if y_real[i]==y_predicted[i]:
            pass
        else:
            #add to the number of images we will display
            p+=1
        
        #ready for the next image
        i+=1
    #re inits some of the variables
    #definitely a sign of good code :)
    i=0
    j=1

    #creates a matplotlib plot with the images on
    for el in x_data:
        #if the ith prediction was correct do nothing
        if y_real[i]==y_predicted[i]:
            pass
        else:
            #create the jth subplot that is a size that means we can fit a square of images on based on the previous code that proabably should have in a subroutine but oh well
            figure.add_subplot(ceil(sqrt(p)),ceil(sqrt(p)),j)
            #add the ith image to the subplot
            plt.imshow(x_data[i])

            #add a label of what was predicted in case we as humans can't figure out what the image is
            if y_predicted[i]==0:
                plt.title("Predicted: Cat Actual: Dog",fontsize=6)
            else:
                plt.title("Predicted: Dog Actual: Cat",fontsize=6)
            #ready for the next subplot
            j+=1
        #ready for the next image
        i+=1    
    #show the entire figure with all the images on
    plt.show()

visualize_incorrect_labels(X, Y, np.array(test_predictions))
