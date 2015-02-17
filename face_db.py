import sqlite3

conn = sqlite3.connect('face.db')
c = conn.cursor()

c.execute('''CREATE TABLE images(
	url TEXT, 
  low_resolution_url TEXT,
  thumbnail_url TEXT,
  users_in_photo BLOB,
  tags TEXT,
  lat REAL,
  lng REAL,
  filter TEXT,
  created_time REAL,
  instagram_id TEXT,
  link TEXT,
  username TEXT,
  faces_rois BLOB,
  faces BLOB,
  caption TEXT,
	api_call_lat REAL,
	api_call_lng REAL
	)''')

conn.commit()

c.execute('''CREATE TABLE faces(
  image_table_id INT,
  face_xywh BLOB, 
  eyes_rois BLOB, 
  eyes_xywh_relative BLOB, 
  eyes_xywh_absolute BLOB,
  mouth_rois BLOB, 
  mouth_xywh_relative BLOB,  
  mouth_xywh_absolute BLOB,
  smile_rois BLOB,
  smile_xywh_relative BLOB, 
  smile_xywh_absolute BLOB
  )''')

conn.commit()
conn.close()

