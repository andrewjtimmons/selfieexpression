import sqlite3

conn = sqlite3.connect('face.db')
c = conn.cursor()

c.execute('''CREATE TABLE images(
	url TEXT, 
  low_resolution_url TEXT,
  thumbnail_url TEXT,
  users_in_photo TEXT,
  tags TEXT,
  lat REAL,
  lng REAL,
  filter TEXT,
  created_time TEXT,
  instagram_id TEXT,
  link TEXT,
  username TEXT,
  faces TEXT,
  caption TEXT,
	api_call_lat REAL,
	api_call_lng REAL
	)''')

conn.commit()

c.execute('''CREATE TABLE faces(
  image_table_id INT,
  url TEXT,
  face_xywh TEXT, 
  eyes_xywh_relative TEXT, 
  eyes_xywh_absolute TEXT,
  mouth_xywh_relative TEXT,  
  mouth_xywh_absolute TEXT,
  smile_xywh_relative TEXT, 
  smile_xywh_absolute TEXT
  )''')

conn.commit()
conn.close()

