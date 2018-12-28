# catvdogcnn
A CNN that uses transfer learning on mobilenet to achieve high accuracy in categorising whether an image contains a cat or a dog

WARNING ONLY LINUX IS OFFICIALLY SUPPORTED

Feedback and pull requests are very welcome


## Copyright
The [getdata.py](getdata.py) code is modified from this https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/  
The image URLs are from imagenet

Copyright © 2018  Rory Sharp All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU General Public License
along with this program.  If you have not received this, see <http://www.gnu.org/licenses/gpl-3.0.html>.

For a summary of the licence go to https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)

## Prerequisites
### For One Liner
* Curl `apt-get install curl`
### For Manual Install
* [Python 3](https://www.python.org/downloads/)
* Keras `pip3 install keras`
* Numpy `pip3 install numpy`
* TensorFlow `pip3 install tensorflow`
* Scikit-Learn `pip3 install sklearn`
* h5py `pip3 install h5py`
* PIL`pip3 install Pillow`
* libhdf5 (only needed on some systems) `sudo apt-get install libhdf5-serial-dev`
## Retraining One Liner Install (Not tested yet!)
`curl https://raw.githubusercontent.com/qwertpi/catvdogcnn/master/rtinstall.bash | sudo bash`

## Prediction One Liner Install (Not tested yet!)
`curl https://raw.githubusercontent.com/qwertpi/catvdogcnn/master/prinstall.bash | sudo bash`

### Manual Installation
1\. Click the green button labelled clone or download

2\. Choose download zip

3\. Save the zip file and unzip it

## Retraining (`Bash commands in code text`) (for training with data augmentation follow the next set of steps NOT these ones)
1\. Make a folder called images and then a folder called dogs and cats inside it `mkdir images && mkdir images/cats && mkdir images/dogs`  
2a\. Run getdata.py on the dogs URLs into the images/dogs folder `python3 getdata.py -u dogs.txt -o images/dogs/`  
2b\. Run getdata.py on the cats URLs `python3 getdata.py -u cats.txt -o images/cats/`  
3a\. Run `cd images/cats/ && for file in *; do convert $file -resize 64x64 -gravity center -background "rgb(0,0,0)" -extent 64x64 $file; done` (only works on Linux, Windows users will need to find a way to batch resize images to 64 by 64 pixels filled with black to make it exactly 64 by 64)  
3b\. Run `cd images/dogs/ && for file in *; do convert $file -resize 64x64 -gravity center -background "rgb(0,0,0)" -extent 64x64 $file; done`  
4\. Download the mobilenet-1-224 weights. I did this by changing the bit of lines 311 and 317 of /usr/local/lib/python3.5/dist-packages/keras_applications/mobilenet.py that says 'mobilenet_%s_%d' to read 'mobilenet_%s_224'. Your mileage, line numbers and file paths may vary.  
5\. Run train.py
## Data augmentation (optional)
1\. Make a folder called images and then a folder called dogs and cats inside it `mkdir images && mkdir images/cats && mkdir images/dogs`  
2a\. Run getdata.py on the dogs URLs into the images/dogs folder `python3 getdata.py -u dogs.txt -o images/dogs/`  
2b\. Run getdata.py on the cats URLs `python3 getdata.py -u cats.txt -o images/cats/`  
3a\. Run aug.py to create new images with added noise and random brighnes and cropping  
3b\. (Optional) Copy your old images to augmented as well `cp images/cats/* augmented/cats/ && cp images/dogs/* augmented/dogs/`  
3a\. Run `cd augmented/cats/ && for file in *; do convert $file -resize 64x64 -gravity center -background "rgb(0,0,0)" -extent 64x64 $file; done` (only works on Linux, Windows users will need to find a way to batch resize images to 64 by 64 pixels filled with black to make it exactly 64 by 64)  
3b\. Run `cd augmented/dogs/ && for file in *; do convert $file -resize 64x64 -gravity center -background "rgb(0,0,0)" -extent 64x64 $file; done`  
4\. Download the mobilenet-1-224 weights. I did this by changing the bit of lines 311 and 317 of /usr/local/lib/python3.5/dist-packages/keras_applications/mobilenet.py that says 'mobilenet_%s_%d' to read 'mobilenet_%s_224'. Your mileage, line numbers and file paths may vary.  
5\. Run generator train.py  
## Predicting
Change the file variable in predict.py to point towards the image you want to check then run it
## Tweaking
* To use your own images of cats and dogs change the URLs in cats.txt and dogs.txt or put your images into the cats and dogs folder
* In train.py putting batch_size as high as your ram can handle should improve performance
* For distinguishing between two classes other than cats and dogs do the same as using your own images of cats and dogs but you will probably want to swap images/cats and images/dogs for your class names. You will also need to switch the directories [here](https://github.com/qwertpi/catvdogcnn/blob/8fcb2fd3fa410c1363a1298db40cb98bbbbff5ee/train.py#L43) in train.py or [here](https://github.com/qwertpi/catvdogcnn/blob/8fcb2fd3fa410c1363a1298db40cb98bbbbff5ee/generator%20train.py#L79) in generator train.py and [here](https://github.com/qwertpi/catvdogcnn/blob/master/aug.py#L33-L34), [here](https://github.com/qwertpi/catvdogcnn/blob/master/aug.py#L39) and [here](https://github.com/qwertpi/catvdogcnn/blob/8fcb2fd3fa410c1363a1298db40cb98bbbbff5ee/aug.py#L45) in aug.py
