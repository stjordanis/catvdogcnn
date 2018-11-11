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
            im = Image.open(requests.get(re.sub('farm..','',url), stream=True).raw)
            im.save(args["output"]+"{}.png".format(str(total).zfill(8)))
            total += 1
            if total%10==0:
                print(total)
	# handle if any exceptions are thrown during the download process
	except:
	    print("The image could not be found!")
print(total)
