#Selfie Expression

###A program to
1.  Make an api call to instragram to get photos from a geographic location
2.  Attempt to detect faces in the images
3.  Store the metadata in sqlite3

###Example call from command line:
python face.py -l 40.7359 -g -73.9903086 -m [CURRENT TIMESTAMP] -t [TIME STAMP OF HOW FAR BACK YOU WANT TO GO] -c [YOUR_CLIENT_ID]


Currently it will pull ten images for each ten minute block between -m and -t.

You can play with those settings by changing:
self.num_photos = 10
in the API_call() class
or 
changing the 600 seconds to something else in:
return new_max_timestamp - 600
in the def get_new_max_timestamp

You may get slightly less than the number of images you expect because it excludes video content.  
I got about 1250 images when attempting to pull 1440.

Have fun.
