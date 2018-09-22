from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from os import system as bash
from time import time
file="/some/file/path"
command="convert "+file+' -resize 64x64 -gravity center -background "rgb(0,0,0)" -extent 64x64 '+file
bash(command)
img = image.load_img(file, target_size=None)
img = np.array(img)
img=np.array([img/255])
model = load_model('model.h5')
prediction=model.predict(img)
if prediction.argmax()==0:
    print("Cat with",prediction[0]*100+"%","certianty)
elif prediction.argmax()==1:
    print("Dog with",prediction[1]*100+"%","certianty)
else:
    print("Well this is awkard. This isn't meant to happen. Please submit an issue on GitHub with the following information")
    print(prediction)
    print(command)
    print(file)
