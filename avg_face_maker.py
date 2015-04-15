"""Create an average face from a list of faces"""

import numpy as np
import cv2
import sqlite3
import json

conn = sqlite3.connect("face.db")
c = conn.cursor()

faces_data = c.execute("SELECT * FROM (SELECT * FROM faces) as t1 inner join (select rowid, instagram_id from images) as t2 on t1.image_table_id = t2.rowid").fetchall()

ids_and_rois = [(x[10], x[2]) for x in faces_data]
cropped_faces = np.zeros((640,640))
for ig_id, face_roi in ids_and_rois:
    img = cv2.imread('grayscale_images/'+ig_id+".jpg",0)
    face_roi = json.loads(face_roi)
    if img is not None:
        cropped_img = img[face_roi[1]:face_roi[1]+face_roi[3], face_roi[0]:face_roi[0]+face_roi[2]]
        resized_image = cv2.resize(cropped_img, (640, 640))  
        cropped_faces += resized_image
        
avg_face = cropped_faces/len(ids_and_rois)
avg_face = avg_face.astype(np.uint8)
cv2.imshow('img',avg_face)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("avg_face_from_numpy_large"+".jpg", avg_face)