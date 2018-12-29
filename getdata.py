#This code is modified from https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/

import argparse
import requests
from PIL import Image
import re

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True,
    help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True,
    help="path to output directory of images")
args = vars(ap.parse_args())

# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
rows = open(args["urls"],encoding="utf-8").read().strip().split("\n")
total = 0

# loop the URLs
for url in rows:
    try:
        #use requests to get the raw data of the url and open it in PIL
        #re.sub is because removing the farm portion from static.flickr.com addresses makes them more likely to not 404
            im = Image.open(requests.get(re.sub('farm..','',url), stream=True).raw)
        #save the image with each image being concetivly named going 00000000.png then 00000001.png
            im.save(args["output"]+"{}.png".format(str(total).zfill(8)))
        #add 1 to the total variable
            total += 1
        #every mutliple of 10 print the total
            if total%10==0:
                print(total)
    # handle if any exceptions are thrown during the download process
    except:
        print("The image could not be found!")
#print total at the end of running
print(total)
