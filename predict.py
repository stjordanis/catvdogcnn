from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from subprocess import check_output as bash
from time import time
if "tmpfs" in bash("df -T /tmp",shell=True).decode("utf-8"):
    print("Your system supports writing to RAM so image loading should be fast :)")
else:
    print("Your /tmp directory doesn't use RAM. This isn't an error it just means resizing the image will be a bit slower")
    
from os import system as bash

file="/some/file/path"
if file[-1]=="/":
    file[:-1]

command="convert "+file+' -resize 64x64 -gravity center -background "rgb(0,0,0)" -extent 64x64 '+"/tmp/"+file.split("/")[-1]
file="/tmp/"+file.split("/")[-1]
bash(command)

img = image.load_img(file, target_size=None)
img = np.array(img)
img=np.array([img/255])
model = load_model('model.h5')

prediction=model.predict(img)

if prediction.argmax()==0:
    print("Cat with",str(prediction[0][0]*100)+"%","certianty")
elif prediction.argmax()==1:
    print("Dog with",str(prediction[0][1]*100)+"%","certianty")
else:
    print("Well this is awkard. This isn't meant to happen. Please submit an issue on GitHub with the following information")
    print(prediction)
    print(command)
    print(file)
    print(bash("df -T /tmp",shell=True).decode("utf-8"))
    
command="rm "+file
bash(command)
