"""A program to
1.  Make an api call to instragram to get photos from a geographic location
2.  Attempt to detect faces in the images
3.  Store the metadata in sqlite3

Example call from command line is:
python face.py -l 40.7359 -g -73.9903086 -m 1424235599 -t 1424149199 -c [YOUR_CLIENT_ID]

"""

import numpy as np
import pandas as pd
import cv2
import os

import json
from PIL import Image
import urllib2
import cStringIO

import time
import sys 
import getopt
import sqlite3

from socket import error as SocketError

# for the casecades to work you need install CV2 and point these to your local dir
FACE_CASECADE = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
EYE_CASECADE = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_eye.xml')
MOUTH_CASECADE = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
SMILE_CASECADE = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_smile.xml')

class API_call():
  """ Makes api calls to instragram's media search endpoint"""
  def __init__(self, lat, lng, max_timestamp, client_id):
    self.lat = lat
    self.lng = lng
    self.max_timestamp = max_timestamp
    self.client_id = client_id
    self.num_photos = 10
    self.api_endpoint =  "https://api.instagram.com/v1/media/search?lat=%s&lng=%s&max_timestamp=%s&distance=5000&client_id=%s&count=%s" % (self.lat, self.lng, self.max_timestamp, self.client_id, self.num_photos)
    self.response =  urllib2.urlopen(self.api_endpoint)
    self.data = json.load(self.response)


class Img():
  """ Object created for each image

  Input Variables
  entry:  the api call data for a specific image
  api_call_lat: the latitude of where the api call was centered
  api_call_lng: the longitude of where the api call was centered

  Class Variables:
  self.url: url of image
  self.low_resolution_url: low res url of image
  self.thumbnail_url: thumbnail url of image
  self.users_in_photo: users tagged in the photo
  self.tags: hashtages on the photo
  self.lat: lat of the photo
  self.lng: lng of the photo
  self.filter: filter applied to photo
  self.created_time:  timestamp of the photo creation
  self.id: instagram's id of the photo
  self.link:  link to the instagram front end web interface of the photo
  self.username: owner of photo's username
  self.color_image: actual color image represented in numpy array
  self.grayscale_image: actual color image represented in numpy array
  self.faces_rois:  numpy array of grayscale image that might be a face
  self.faces: collection of four points that bound a potential face in a box that is generated from the haar casecade.
  self.num_faces: len(self.faces_rois)
  self.caption: caption on photo by user
  self.api_call_lat: the latitude of where the api call was centered
  self.api_call_lng: the longitude of where the api call was centered 
  """

  def __init__(self, entry, api_call_lat, api_call_lng):
    self.url = entry['images']['standard_resolution']['url']
    self.low_resolution_url = entry['images']['low_resolution']['url']
    self.thumbnail_url = entry['images']['thumbnail']['url']
    if entry['users_in_photo'] != []:
      self.users_in_photo = entry['users_in_photo']
    else: 
      self.users_in_photo = None
    self.tags = entry['tags']
    self.lat = entry['location']['latitude']
    self.lng = entry['location']['longitude']
    self.filter = entry['filter']
    self.created_time = entry['created_time']
    self.id = entry['id']
    self.link = entry['link']
    self.username = entry['user']['username']
    self.color_image = self._create_opencv_image_from_url()
    self.grayscale_image = self._create_grayscale_image()
    self.faces_rois, self.faces = self._detect_faces()
    self.num_faces = len(self.faces_rois)
    try:
      self.caption = entry['caption']['text']
    except TypeError:
      self.caption = ""
    self.api_call_lat = api_call_lat
    self.api_call_lng = api_call_lng
  
  def _create_opencv_image_from_url(self, cv2_img_flag = 1):
    """ Get image from URL and convert to an openCV image."""
    request = urllib2.urlopen(self.url)
    img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)
  
  def _create_grayscale_image(self):
    """ Turn color image into grayscale."""
    return cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)

  def _detect_faces(self):
    """ Detect faces in the image.  Returns an empty list if no faces. """
    faces = FACE_CASECADE.detectMultiScale(self.grayscale_image, scaleFactor = 1.05, minNeighbors = 3)
    faces_rois = []
    for (x,y,w,h) in faces:
      self._draw_rectangle(self.color_image, (x,y), (x+w,y+h), (255,0,0))
      faces_rois.append(self.grayscale_image[y:y+h, x:x+w])
    return faces_rois, faces
  
  def _draw_rectangle(self, image, pt1, pt2, color, thickness = 2):
    """ Draw rectangle around a region of interests with a arbitrary color. """
    cv2.rectangle(image, pt1, pt2, color, thickness)
      
  def show_color_image(self):
    """ Display image on screen and close on key press. """
    cv2.imshow('img',self.color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
  def show_grayscale_image(self):
    """ Display image on screen and close on key press. """
    cv2.imshow('img',self.grayscale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
      
  def has_faces(self):
    """ Returns True if more than one face is detected. """
    if len(self.faces_rois) > 0:
      return True
    return False
      
      
class Face():
  """ Object for a detected face region of interest in an img object.  
  Each image has 0 or more faces.
  
  Input variables
  url: url of original image
  color_image: the color image from the url
  grayscale_image: the grayscale image from the url
  face_roi: numpy array of grayscale image that might be a face
  face_xywh: the four points that bound a this specific potential face in a box that is generated from the haar casecade.  Comes from self.faces in Img class. 

  Class variables:
  self.eyes_rois: numpy array of grayscale image that might be eyes
  self.eyes_xywh_relative: position of eyes relative to face_roi
  self.eyes_xywh_absolute: position of eyes in relation to the whole image
  self.roi_gray_below_eyes: area below the eyes to look for a mouth in grayscale
  self.roi_color_below_eyes: area below the eyes to look for a mouth in color
  self.mouth_rois: numpy array of grayscale image that might be a mouth
  self.mouth_xywh_relative: position of mouth relative to face_roi
  self.mouth_xywh_absolute: position of mouth in relation to the whole image
  self.smile_rois: numpy array of grayscale image that might be a smile
  self.smile_xywh_relative: position of smile relative to face_roi
  self.smile_xywh_absolute: position of smile in relation to the whole image
  """
  def __init__(self, url, color_image, grayscale_image, face_roi, face_xywh):
    self.url = url
    self.color_image = color_image
    self.grayscale_image = grayscale_image
    self.face_roi = face_roi
    self.face_xywh = face_xywh
    self.eyes_rois, self.eyes_xywh_relative, self.eyes_xywh_absolute = self._detect_eyes()
    if self.has_two_eyes():
      self.roi_gray_below_eyes, self.roi_color_below_eyes = self._create_roi_below_eyes()
      self.mouth_rois, self.mouth_xywh_relative, self.mouth_xywh_absolute = self._detect_mouths()
      self.smile_rois, self.smile_xywh_relative, self.smile_xywh_absolute = self._detect_smiles()
    
  def _detect_eyes(self):
    """ Detect eyes in the image.  Returns an empty list if no eyes. """

    eyes_xywh_relative = EYE_CASECADE.detectMultiScale(self.face_roi, scaleFactor = 1.05, minNeighbors = 3)
    eyes_rois = []
    eyes_xywh_absolute = []
    for x, y, w, h in eyes_xywh_relative:
      self._draw_rectangle(self.face_roi, (x,y), (x+w,y+h), (0,255,0))
      x1, x2, y1, y2 = self._eye_math(x,y,w,h)
      eyes_rois.append(self.color_image[y1:y2, x1:x2])
      eyes_xywh_absolute.append([x1, x2, y1, y2])
    return eyes_rois, eyes_xywh_relative, eyes_xywh_absolute
  
  def _create_roi_below_eyes(self):
    """ Takes the face roi and limits it to the region below the eyes.  This will let
    the mouth casecades just search in that region instead of looking at the whole face.
    Get x,y coords with width and height. """
    x_face, y_face, w_face, h_face = self.face_xywh
    y_eyes = y_face + self.eyes_xywh_relative[0][1] + int(self.eyes_xywh_relative[0][3]*1.5)
    face_bottom = h_face - self.eyes_xywh_relative[0][1] - int(self.eyes_xywh_relative[0][3]*1.5)
    roi_gray_below_eyes = self.grayscale_image[y_eyes:y_eyes+face_bottom, x_face:x_face+w_face]
    roi_color_below_eyes = self.color_image[y_eyes:y_eyes+face_bottom, x_face:x_face+w_face]
    return roi_gray_below_eyes, roi_color_below_eyes

  def _detect_mouths(self):
    """ Detect mouth in the image.  Returns an empty list if no mouth. """
    mouth_xywh_relative = MOUTH_CASECADE.detectMultiScale(self.roi_gray_below_eyes, scaleFactor = 1.05, minNeighbors = 3)
    mouth_rois = []
    mouth_xywh_absolute = []
    for x,y,w,h in mouth_xywh_relative:
      self._draw_rectangle(self.roi_color_below_eyes, (x,y), (x+w,y+h), (0,0,0))
      x1, x2, y1, y2 = self._mouth_math(x,y,w,h)
      self._draw_rectangle(self.color_image, (x1,y1), (x2,y2), (255,255,0), thickness = 1)
      mouth_rois.append(self.color_image[y1:y2, x1:x2])
      mouth_xywh_absolute.append([x1, x2, y1, y2])
    return mouth_rois, mouth_xywh_relative, mouth_xywh_absolute

  def _detect_smiles(self):
    """ Detect smile in the image.  Returns an empty list if no mouth. """
    smile_xywh_relative = SMILE_CASECADE.detectMultiScale(self.roi_gray_below_eyes, scaleFactor = 1.05, minNeighbors = 3)
    smile_rois = []
    smile_xywh_absolute = []
    for x , y, w, h in smile_xywh_relative:
      self._draw_rectangle(self.roi_color_below_eyes, (x,y), (x+w,y+h), (255,255,255))
      x1, x2, y1, y2 = self._mouth_math(x,y,w,h)
      self._draw_rectangle(self.color_image, (x1,y1), (x2,y2), (255,255,0), thickness = 1)
      smile_rois.append(self.color_image[y1:y2, x1:x2])
      smile_xywh_absolute.append([x1, x2, y1, y2])
    return smile_rois, smile_xywh_relative, smile_xywh_absolute
  
  def _eye_math(self, x, y, w, h):
    """ Returns points from the eye roi that are in context of the whole image, not just the eye roi """
    x1 = self.face_xywh[0]+x
    x2 = self.face_xywh[0]+x+w
    y1 = self.face_xywh[1]+y
    y2 = self.face_xywh[1]+y+h
    return x1, x2, y1, y2
  
  def _mouth_math(self, x, y, w, h):
    """ Returns points from the mouth roi that are in context of the whole image, not just the mouth roi """
    x1 = self.face_xywh[0]+x
    x2 = self.face_xywh[0]+x+w
    y1 = self.face_xywh[1]+self.eyes_xywh_relative[0][1]+int(self.eyes_xywh_relative[0][3]*1.5)+y
    y2 = self.face_xywh[1]+self.eyes_xywh_relative[0][1]+int(self.eyes_xywh_relative[0][3]*1.5)+y+h
    return x1, x2, y1, y2

  def _draw_rectangle(self, image, pt1, pt2, color, thickness = 2):
    """ Draw rectangle around a region of interests with a arbitrary color. """
    cv2.rectangle(image, pt1, pt2, color, thickness)
  
  def show_color_image(self):
    """ Display image on screen and close on key press. """
    cv2.imshow('img',self.color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  def show_grayscale_image(self):
    """ Display image on screen and close on key press. """
    cv2.imshow('img',self.grayscale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  def has_two_eyes(self):
    """ Returns True if two eyes are detected. """
    if len(self.eyes_rois) == 2:
      return True
    return False    

  def has_one_mouth(self):
    """ Returns true if only one mouth. """
    if len(self.mouth_rois) == 1:
      return True
    return False

  def has_zero_or_one_smile(self):
    """ Returns true if it has zero or one smiles. """
    if len(self.smile_rois) <= 1:
      return True
    return False


def main(argv):
  """The main function that strings together the classes and the non class methods.
  It does the following:
  1.  Make an api call to instragram to get photos from a geographic location
  2.  Attempt to detect faces in the images
  3.  Store the metadata in sqlite3

  Example call from command line is:
  python face.py -l 40.7359 -g -73.9903086 -m 1424235599 -t 1424149199 -c [YOUR_CLIENT_ID]

  Input variables passed via flags
  api_call_lat: the latitude of where the api call was centered
  api_call_lng: the latitude of where the api call was centered
  max_timestamp: The largest possible timestamp you want an image to have. (note instagram api does not always respect this precisely, but it is usually pretty close) 
  min_timestamp: The smallest possible timestamp you want an image to have. (note instagram api does not always respect this precisely, but it is usually pretty close) 
  client_id:  Your instagram api client id

  This function has time.sleep() in a few places since you have to call the images from the web.  The api only gives you the metadata and we dont want to slam their servers. 
  """
  api_call_lat, api_call_lng, max_timestamp, min_timestamp, client_id = parse_cmd_args(argv)
  calls_made = 0
  face_count = 0
  t1 = time.time()
  conn, cursor = db_connect()
  already_in_db = cursor.execute("SELECT instagram_id from images").fetchall()
  processed_images = set([str(x[0]) for x in already_in_db])
  socket_error_ids = []
  #for call in range(num_api_calls):
  while max_timestamp > min_timestamp:
    images = get_image_entries_from_api(api_call_lat, api_call_lng, max_timestamp, client_id)
    for entry in images:
      try:
        img = Img(entry, api_call_lat, api_call_lng)
        
        if is_new_image(img.id, processed_images):
          print img.url + "\n" + img.created_time   
          image_table_id = insert_in_image_db_and_return_id(img, cursor)
  
          for possible_face, face_xywh in zip(img.faces_rois, img.faces):
            face = Face(img.url, img.color_image, img.grayscale_image, possible_face, face_xywh)
            if face.has_two_eyes() and face.has_one_mouth() and face.has_zero_or_one_smile():
                print "face_found"
                insert_in_face_db(face, image_table_id, img.url, cursor)
                face_count += 1
          conn.commit()  
          print "commited to db"
          save_image_to_disk(img)

      except cv2.error:
        continue 

      except SocketError as E:
        print "socket error, connection reset by peer, pausing for 3 minutes"
        time.sleep(180)
        socket_error_ids.append(entry['id'])

      time.sleep(1)
    
    max_timestamp = get_new_max_timestamp(max_timestamp, img.created_time)
    calls_made += 1
    print str(face_count) + " faces found through loop " + str(calls_made)
    print "pausing for 3 seconds"
    time.sleep(3)
  
  conn.close()
  print 'time taken is ' + str(time.time() - t1)
  print "the following images failed due to socket hangups"
  print socket_error_ids

def parse_cmd_args(argv):
  """ Turns command line flags into variables """
  try:
      opts, args = getopt.getopt(argv,"hl:g:m:t:c:",["lat=", "lng=", "max_timestamp=", "min_timestamp=", "client_id="])
  except getopt.GetoptError:
    print 'face.py face.py "-l --lat, -g --lng, -m --max_timestamp, -c --client_id="'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
       print 'face.py "-l --lat, -g --lng, -m --max_timestamp, -c --client_id="'
       sys.exit()
    elif opt in ("-l", "--lat"):
       api_call_lat = float(arg)
    elif opt in ("-g", "--lng"):
       api_call_lng = float(arg)
    elif opt in ("-m", "--max_timestamp"):
       max_timestamp = int(arg)
    elif opt in ("-t", "--min_timestamp"):
       min_timestamp = int(arg)
    elif opt in ("-c", "--client_id"):
       client_id = str(arg)
    else:
      assert False, "unhandled option"

  return api_call_lat, api_call_lng, max_timestamp, min_timestamp, client_id

def db_connect():
  """ Connect to sqlite db. """
  conn = sqlite3.connect('face.db')
  cursor = conn.cursor()
  return conn, cursor

def get_image_entries_from_api(api_call_lat, api_call_lng, max_timestamp, client_id):
  """ Gets images from the api """
  response = API_call(lat = api_call_lat, lng = api_call_lng, max_timestamp = max_timestamp, client_id = client_id)
  images = [entry for entry in response.data['data'] if entry['type'] == 'image']
  return images

def insert_in_image_db_and_return_id(img, cursor):
  """ Insert the attributes into the db
  Vars like img.faces have to have some conversion because cv2 haar cascades
  return either numpy.ndarray or a empty ().  So we convert ()) to a list
  so they all have the same datatype.
  Retuns the key of the image for storing in the faces db for joins later. """
  if img.faces != ():
    faces_for_db = img.faces.tolist()
  else: 
    faces_for_db = []
  row = [
    img.url, 
    img.low_resolution_url,
    img.thumbnail_url,
    json.dumps(img.users_in_photo),
    json.dumps(img.tags),
    img.lat,
    img.lng,
    img.filter,
    img.created_time,
    img.id,
    img.link,
    img.username,
    json.dumps(faces_for_db),
    img.caption,
    img.api_call_lat,
    img.api_call_lng
  ]
  cursor.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
  return cursor.lastrowid

def insert_in_face_db(face, image_table_id, url, cursor):
  """ Insert the attributes into the db
  variables with relative are numpy arrays
  variables with abosulte are a python array.  
  absolute variables should not need conversion but was failing 
  with error TypeError: 351 is not JSON serializable.  
  However it works in terminal. So we convert the 
  list to a numpy array and then back to a list. Hackey workaround for now.
  """

  eyes_xywh_absolute = np.asarray(face.eyes_xywh_absolute)
  mouth_xywh_absolute = np.asarray(face.mouth_xywh_absolute)
  smile_xywh_absolute = np.asarray(face.smile_xywh_absolute)
  if face.smile_xywh_relative != ():
    smile_xywh_relative = face.smile_xywh_relative.tolist()
  else: 
    smile_xywh_relative = []
  row = [
    image_table_id,
    url,
    json.dumps(face.face_xywh.tolist()),
    json.dumps(face.eyes_xywh_relative.tolist()), 
    json.dumps(eyes_xywh_absolute.tolist()),
    json.dumps(face.mouth_xywh_relative.tolist()),  
    json.dumps(mouth_xywh_absolute.tolist()),
    json.dumps(smile_xywh_relative), 
    json.dumps(smile_xywh_absolute.tolist())
  ]
  cursor.execute("INSERT INTO faces VALUES (?,?,?,?,?,?,?,?,?)", row)
 
def is_new_image(image_id, processed_images):
  """ Checks if image has already been processed since instagram api sometimes 
  does not respect max_timestamp.  See 
  http://stackoverflow.com/questions/23792774/instagram-api-media-search-endpoint-not-respecting-max-timestamp-parameter
  and http://stackoverflow.com/questions/25155620/instagram-api-media-search-max-timestamp-issue
  """

  if image_id in processed_images:
    print 'seen image_id %s before, skipping it' % image_id
    return False
  processed_images.add(image_id)
  return True

def save_image_to_disk(img):
  """Save image to local storage. """
  cv2.imwrite("images/" + img.id + ".jpg",img.color_image)
  cv2.imwrite("grayscale_images/" + img.id + ".jpg", img.grayscale_image)

def get_new_max_timestamp(last_max_timestamp, current_image_timestamp):
  """ Gives the smaller of the last max timestamp vs the last image's timestamp.  
  Then it subtracts 600 seconds from that.  This is needed because the instagram
  API does not always respect max_timestamp and you could get stuck in a loop 
  where all photos have a max_timestamp greater than what was set
  """
  new_max_timestamp = min(last_max_timestamp, current_image_timestamp)
  return new_max_timestamp - 600

if __name__ == '__main__':
    main(sys.argv[1:])