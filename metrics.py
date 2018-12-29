#code modified from https://medium.com/@kylepob61392/airplane-image-classification-using-a-keras-cnn-22be506fdb53
#this code is a mess so I'm not going to even bother to try to comment it

from keras.models import load_model
import numpy as np
from os import listdir
from keras.preprocessing import image
from math import sqrt, ceil

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
    y_data.append([0])
for i in range(0,132):
    y_data.append([1])
    
model=load_model("model.h5")
x_data=["images/cats/","images/dogs/"]
Y = np.array(y_data)
X = np.array(load_images(x_data))
test_predictions=[]
predictions = model.predict(X)
for el in predictions:
    test_predictions.append(el.argmax())
    
import matplotlib.pyplot as plt
def visualize_incorrect_labels(x_data, y_real, y_predicted):
    figure = plt.figure()
    i=0
    j=1
    p=0
    for el in x_data:
        if y_real[i]==y_predicted[i]:
            pass
        else:
            p+=1
        i+=1
        
    i=0
    j=1
    for el in x_data:
        if y_real[i]==y_predicted[i]:
            pass
        else:
            print(y_real[i])
            print(y_predicted[i])
            figure.add_subplot(ceil(sqrt(p)),ceil(sqrt(p)),j)
            plt.imshow(x_data[i])
            if y_predicted[i]==0:
                plt.title("Predicted: Cat Actual: Dog",fontsize=6)
            else:
                plt.title("Predicted: Dog Actual: Cat",fontsize=6)
            j+=1
        i+=1    
    plt.show()

visualize_incorrect_labels(X, Y, np.array(test_predictions))

