from imgaug import augmenters as iaa
from keras.preprocessing import image
import numpy as np
from os import listdir
from PIL import Image
total=0

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
            imgs.append(img)
            #delete the img variable to save RAM
            del img
    return imgs

seq = iaa.Sequential([
      iaa.Multiply((0.5, 1.5)),
      iaa.AdditiveGaussianNoise(scale=(0, 0.075*255),per_channel=True),
      iaa.Crop(percent=(0, 0.3)), #0.3=30%
      iaa.Fliplr(0.5)
      ],random_order=True)

cats=np.array(load_images(["images/cats/"]))
dogs=np.array(load_images(["images/dogs/"]))
for i in range(0,5):
    cats_aug = seq.augment_images(cats)
    for img in cats_aug:
        im = Image.fromarray(img)
        im.save("augmented/cats/aug_"+"{}.png".format(str(total).zfill(8)))
        total+=1

total=0
for i in range(0,5):
    dogs_aug = seq.augment_images(dogs)
    for img in dogs_aug:
        im = Image.fromarray(img)
        im.save("augmented/dogs/aug_"+"{}.png".format(str(total).zfill(8)))
        total+=1

