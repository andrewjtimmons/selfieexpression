"""
Creates a window with a slider that 
"""

#TODO remove unused
import numpy as np
import pandas as pd
import cv2
from sklearn import datasets
import os

import json
from PIL import Image
import urllib2
import cStringIO

import time
import sys 
import getopt
import sqlite3


def main():
  conn, cursor = db_connect()
  unprocessed = cursor.execute("SELECT instagram_id, rowid FROM images WHERE faces_actual IS NULL").fetchall()
  failed_images = []
  images_to_look_at = [(str(x[0]), int(x[1])) for x in unprocessed]
  for instagram_id, rowid in images_to_look_at:
    print instagram_id
    try:
			cv2.namedWindow('img')
			img = cv2.imread("images_boxes/" + instagram_id + ".jpg")
			cv2.createTrackbar('faces_actual','img',0,20, onChange)
			previous_val = 0
			while(1):
				cv2.imshow('img',img)
				k = cv2.waitKey(1) & 0xFF
				if k == 27:
					break

				faces_actual = cv2.getTrackbarPos('faces_actual', 'img')

				if faces_actual == previous_val:
					pass
				else:
					print faces_actual
					previous_val = faces_actual

			cv2.destroyAllWindows()
			cursor.execute("UPDATE images SET faces_actual = %i where rowid = %i" %(faces_actual, rowid))			
			conn.commit()
			print "finished on img %s with faces_actual as %s" % (rowid, faces_actual)
			previous_val = 0
			faces_actual = 0
    except cv2.error:
      failed_images.append(instagram_id)
      continue 

  conn.close()
  if failed_images:
  	print 'failed on these: %s' % failed_images

def onChange(val):
	#required for getTrackbarPos call, but not for this program.
	pass

def db_connect():
  conn = sqlite3.connect('face.db')
  cursor = conn.cursor()
  return conn, cursor

if __name__ == '__main__':
    main()