# Visual test for seeing if absolute boxes are correct.
# You can call eyes, mouth or smile _xywh_absolute and
# see if the boxes line up.  You should see a thick white
# box over the second image in the right region
import face_detector 
img = face_detector.Img('http://scontent-b.cdninstagram.com/hphotos-xfa1/t51.2885-15/10864805_880408875336991_181445591_n.jpg')
for possible_face, face_xywh in zip(img.faces_rois, img.faces):
    face = face_detector.Face(img.url, img.color_image, img.grayscale_image, possible_face, face_xywh)
    if face.has_two_eyes():
        #face.show_color_image()
        (x1,x2,y1,y2) = face.eyes_xywh_absolute[1]
        face.show_color_image()
        face._draw_rectangle(face.color_image, (x1,y1), (x2,y2), (255,255,255), thickness = 4)
        face.show_color_image()


