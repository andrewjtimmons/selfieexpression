 img.tags = entry['tags']
    img.lat = entry['location']['latitude']
    img.lng = entry['location']['longitude']
    img.filter = entry['filter']
    img.created_time = entry['created_time']
    img.id = entry['id']
    img.link = entry['link']
    img.username = entry['user']['username']
    img.color_image = img._create_opencv_image_from_url()
    img.grayscale_image = img._create_grayscale_image()
    img.faces_rois, img.faces = img._detect_faces()
    img.num_faces = len(img.faces_rois)
    try:
      img.caption = entry['caption']['text']
    except TypeError:
      img.caption = ""
    img.api_call_lat = api_call_lat
    img.api_call_lng = api_call_lng
    )")