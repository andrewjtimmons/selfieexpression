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
    )")