# catvdogcnn
A CNN that uses transfer learning on mobilenet to achive high accuarcy in deciding if an image contains a cat or a dog

WARNING ONLY LINUX IS OFFICALY SUPPORTED

Feedback and pull requests are very welcome


## Copyright
The getdata.py code is modified from this https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/  
The image urls are from imagnet

Copyright © 2018  Rory Sharp All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU General Public License
along with this program.  If you have not received this, see <http://www.gnu.org/licenses/gpl-3.0.html>.

For a summary of the licence go to https://tldrlegal.com/license/gnu-lesser-general-public-license-v3-(lgpl-3)

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
## One Liner Install (Linux Only)
Coming soon!
### Manual Installation
1\. Click the green button labelled clone or download

2\. Choose download zip

3\. Save the zip file and unzip it

## Usage (`Bash commands in code text`)
1\. Make a folder called images and then a folder called dogs and cats inside it `mkdir images && mkdir images/cats && mkdir images/dogs`
2a\. Run getdata.py on the dogs urls into the images/dogs folder `python3 getdata.py -u dogs.txt -o images/dogs/` 
2b\. Run getdata.py on the cats urls `python3 getdata.py -u cats.txt -o images/cats/` 
3a\. Run `cd cats/ && for file in *; do convert $file -resize 64x64 -gravity center -background "rgb(0,0,0)" -extent 64x64 $file; done` (only works on linux, windows users will need to find a way to batch resize images to 64 by 64 pixels filled with black to make it exactly 64 by 64)
3b\. Run `cd dogs/ && for file in *; do convert $file -resize 64x64 -gravity center -background "rgb(0,0,0)" -extent 64x64 $file; done`
4\. Download the mobilenet-1-224 weights. I did this by changing the bit of lines 311 and 317 of /usr/local/lib/python3.5/dist-packages/keras_applications/mobilenet.py that says 'mobilenet_%s_%d' to read 'mobilenet_%s_224'. Your milage, line numbers and file paths may vary.
5\. Run transfer learning train.py

## Tweaking
* To use your own images of cats and dogs change the urls in cats.txt and dogs.txt or put your images into the cats and dogs folder
* In transfer learning train.py putting batch_size as high as your ram can handle should improve performance
