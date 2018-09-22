import argparse
import requests
from PIL import Image
import os
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

	# handle if any exceptions are thrown during the download process
	except:
		print("ERROR!")
