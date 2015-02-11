#instagram imports
import bottle
import beaker.middleware
from bottle import route, redirect, post, run, request, hook
from instagram import client, subscriptions
bottle.debug(True)

#python image imports
from PIL import Image
import urllib2
import cStringIO


#cv2 imports
import numpy as np
import pandas as pd
import cv2
from sklearn import datasets
import os

#change dirs on these
face_cascade = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml')
smile_cascade = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_smile.xml')


session_opts = {
    'session.type': 'file',
    'session.data_dir': './session/',
    'session.auto': True,
}

app = beaker.middleware.SessionMiddleware(bottle.app(), session_opts)

CONFIG = {
    'client_id': '6b0e70728ae64949a63d28af03bc5cd8',
    'client_secret': 'a0bbb21dbd4345419962964a64a8af3b',
    'redirect_uri': 'http://localhost:8515/oauth_callback'
}

unauthenticated_api = client.InstagramAPI(**CONFIG)

@hook('before_request')
def setup_request():
    request.session = request.environ['beaker.session']

def process_tag_update(update):
    print(update)

reactor = subscriptions.SubscriptionsReactor()
reactor.register_callback(subscriptions.SubscriptionType.TAG, process_tag_update)

@route('/')
def home():
    try:
        url = unauthenticated_api.get_authorize_url(scope=["likes","comments"])
        return '<a href="%s">Connect with Instagram</a>' % url
    except Exception as e:
        print(e)

def get_nav(): 
    nav_menu = ("<h1>Python Instagram</h1>"
                "<ul>"
                    "<li><a href='/recent'>User Recent Media</a> Calls user_recent_media - Get a list of a user's most recent media</li>"
                    "<li><a href='/user_media_feed'>User Media Feed</a> Calls user_media_feed - Get the currently authenticated user's media feed uses pagination</li>"              
                    "<li><a href='/location_recent_media'>Location Recent Media</a> Calls location_recent_media - Get a list of recent media at a given location, in this case, the Instagram office</li>"
                    "<li><a href='/media_search'>Media Search</a> Calls media_search - Get a list of media close to a given latitude and longitude</li>"
                    "<li><a href='/media_popular'>Popular Media</a> Calls media_popular - Get a list of the overall most popular media items</li>"
                    "<li><a href='/user_search'>User Search</a> Calls user_search - Search for users on instagram, by name or username</li>"
                    "<li><a href='/user_follows'>User Follows</a> Get the followers of @instagram uses pagination</li>"
                    "<li><a href='/location_search'>Location Search</a> Calls location_search - Search for a location by lat/lng</li>"      
                    "<li><a href='/tag_search'>Tags</a> Search for tags, view tag info and get media by tag</li>"
                    "<li><a href='/test_search'>Tags</a>loc and img</li>"
                "</ul>")
            
    return nav_menu

@route('/oauth_callback')
def on_callback(): 
    code = request.GET.get("code")
    if not code:
        return 'Missing code'
    try:
        access_token, user_info = unauthenticated_api.exchange_code_for_access_token(code)
        if not access_token:
            return 'Could not get access token'
        api = client.InstagramAPI(access_token=access_token)
        request.session['access_token'] = access_token
        print ("access token="+access_token)
        print api.list_subscriptions()
        print "andy"
    except Exception as e:
        print(e)
    return get_nav()

@route('/recent')
def on_recent(): 
    content = "<h2>User Recent Media</h2>"
    access_token = request.session['access_token']
    if not access_token:
        return 'Missing Access Token'
    try:
        api = client.InstagramAPI(access_token=access_token)
        recent_media, next = api.user_recent_media()
        photos = []
        for media in recent_media:
            photos.append('<div style="float:left;">')
            if(media.type == 'video'):
                photos.append('<video controls width height="150"><source type="video/mp4" src="%s"/></video>' % (media.get_standard_resolution_url()))
            else:
                photos.append('<img src="%s"/>' % (media.get_low_resolution_url()))
            print(media)
            photos.append("<br/> <a href='/media_like/%s'>Like</a>  <a href='/media_unlike/%s'>Un-Like</a>  LikesCount=%s</div>" % (media.id,media.id,media.like_count))
        content += ''.join(photos)
    except Exception as e:
        print(e)              
    return "%s %s <br/>Remaining API Calls = %s/%s" % (get_nav(),content,api.x_ratelimit_remaining,api.x_ratelimit)

@route('/media_like/<id>')
def media_like(id): 
    access_token = request.session['access_token']
    api = client.InstagramAPI(access_token=access_token)
    api.like_media(media_id=id)
    redirect("/recent")

@route('/media_unlike/<id>')
def media_unlike(id): 
    access_token = request.session['access_token']
    api = client.InstagramAPI(access_token=access_token)
    api.unlike_media(media_id=id)
    redirect("/recent")

@route('/user_media_feed')
def on_user_media_feed(): 
    access_token = request.session['access_token']
    content = "<h2>User Media Feed</h2>"
    if not access_token:
        return 'Missing Access Token'
    try:
        api = client.InstagramAPI(access_token=access_token)
        media_feed, next = api.user_media_feed()
        photos = []
        for media in media_feed:
            photos.append('<img src="%s"/>' % media.get_standard_resolution_url())
        counter = 1
        while next and counter < 3:
            media_feed, next = api.user_media_feed(with_next_url=next)
            for media in media_feed:
                photos.append('<img src="%s"/>' % media.get_standard_resolution_url())
            counter += 1
        content += ''.join(photos)
    except Exception as e:
        print(e)              
    return "%s %s <br/>Remaining API Calls = %s/%s" % (get_nav(),content,api.x_ratelimit_remaining,api.x_ratelimit)

@route('/location_recent_media')
def location_recent_media(): 
    access_token = request.session['access_token']
    content = "<h2>Location Recent Media</h2>"
    if not access_token:
        return 'Missing Access Token'
    try:
        api = client.InstagramAPI(access_token=access_token)
        recent_media, next = api.location_recent_media(location_id=514276)
        photos = []
        for media in recent_media:
            photos.append('<img src="%s"/>' % media.get_standard_resolution_url())
        content += ''.join(photos)
    except Exception as e:
        print(e)              
    return "%s %s <br/>Remaining API Calls = %s/%s" % (get_nav(),content,api.x_ratelimit_remaining,api.x_ratelimit)

@route('/media_search')
def media_search(): 
    access_token = request.session['access_token']
    content = "<h2>Media Search</h2>"
    if not access_token:
        return 'Missing Access Token'
    try:
        api = client.InstagramAPI(access_token=access_token)
        media_search = api.media_search(count = 101, lat="40.727184",lng="-73.995833",distance=5000)
        # photos = []
        # for media in media_search:
        #     photos.append('<img src="%s"/>' % media.get_standard_resolution_url())
        # content += ''.join(photos)
        content += str(len(media_search))
        content += "<br />"
        content += str(media_search)
        content += "<br />"
        content += str(dir(media_search[-1]))
    except Exception as e:
        print(e)              
    return "%s %s <br/>Remaining API Calls = %s/%s" % (get_nav(),content,api.x_ratelimit_remaining,api.x_ratelimit)

@route('/media_popular')
def media_popular(): 
    access_token = request.session['access_token']
    content = "<h2>Popular Media</h2>"
    if not access_token:
        return 'Missing Access Token'
    try:
        api = client.InstagramAPI(access_token=access_token)
        media_search = api.media_popular()
        photos = []
        for media in media_search:
            photos.append('<img src="%s"/>' % media.get_standard_resolution_url())
        content += ''.join(photos)
    except Exception as e:
        print(e)              
    return "%s %s <br/>Remaining API Calls = %s/%s" % (get_nav(),content,api.x_ratelimit_remaining,api.x_ratelimit)

@route('/user_search')
def user_search(): 
    access_token = request.session['access_token']
    content = "<h2>User Search</h2>"
    if not access_token:
        return 'Missing Access Token'
    try:
        api = client.InstagramAPI(access_token=access_token)
        user_search = api.user_search(q="Instagram")
        users = []
        for user in user_search:
            users.append('<li><img src="%s">%s</li>' % (user.profile_picture,user.username))
        content += ''.join(users)
    except Exception as e:
        print(e)              
    return "%s %s <br/>Remaining API Calls = %s/%s" % (get_nav(),content,api.x_ratelimit_remaining,api.x_ratelimit)

@route('/user_follows')
def user_follows(): 
    access_token = request.session['access_token']
    content = "<h2>User Follows</h2>"
    if not access_token:
        return 'Missing Access Token'
    try:
        api = client.InstagramAPI(access_token=access_token)
        # 25025320 is http://instagram.com/instagram
        user_follows, next = api.user_follows('25025320')
        users = []
        for user in user_follows:
            users.append('<li><img src="%s">%s</li>' % (user.profile_picture,user.username))
        while next:
            user_follows, next = api.user_follows(with_next_url=next)
            for user in user_follows:
                users.append('<li><img src="%s">%s</li>' % (user.profile_picture,user.username))
        content += ''.join(users)
    except Exception as e:
        print(e)              
    return "%s %s <br/>Remaining API Calls = %s/%s" % (get_nav(),content,api.x_ratelimit_remaining,api.x_ratelimit)

@route('/location_search')
def location_search(): 
    access_token = request.session['access_token']
    content = "<h2>Location Search</h2>"
    if not access_token:
        return 'Missing Access Token'
    try:
        api = client.InstagramAPI(access_token=access_token)
        location_search = api.location_search(lat="40.727184",lng="-73.995833",distance=5000)
        locations = []
        for location in location_search:
            locations.append('<li>%s  <a href="https://www.google.com/maps/preview/@%s,%s,19z">Map</a>  </li>' % (location.name,location.point.latitude,location.point.longitude))
        content += ''.join(locations)
    except Exception as e:
        print(e)              
    return "%s %s <br/>Remaining API Calls = %s/%s" % (get_nav(),content,api.x_ratelimit_remaining,api.x_ratelimit)

@route('/tag_search')
def tag_search(): 
    access_token = request.session['access_token']
    content = "<h2>Tag Search</h2>"
    if not access_token:
        return 'Missing Access Token'
    try:
        api = client.InstagramAPI(access_token=access_token)
        tag_search, next_tag = api.tag_search(q="selfie", count=100)
        tag_recent_media, next = api.tag_recent_media(tag_name=tag_search[0].name)
        photos = []
        for tag_media in tag_recent_media:
            #f = cStringIO.StringIO(urllib2.urllib2(tag_media.get_standard_resolution_url()).read())
            imgURL = tag_media.get_standard_resolution_url()
            LabelFaces(imgURL)
            photos.append('<img src="%s"/>' % tag_media.get_standard_resolution_url())
        content += ''.join(photos)
    except Exception as e:
        print(e)              
    return "%s %s <br/>Remaining API Calls = %s/%s" % (get_nav(),content,api.x_ratelimit_remaining,api.x_ratelimit)

@route('/realtime_callback')
@post('/realtime_callback')
def on_realtime_callback():
    mode = request.GET.get("hub.mode")
    challenge = request.GET.get("hub.challenge")
    verify_token = request.GET.get("hub.verify_token")
    if challenge: 
        return challenge
    else:
        x_hub_signature = request.header.get('X-Hub-Signature')
        raw_response = request.body.read()
        try:
            reactor.process(CONFIG['client_secret'], raw_response, x_hub_signature)
        except subscriptions.SubscriptionVerifyError:
            print("Signature mismatch")

@route('/test_search')
def media_search(): 
    access_token = request.session['access_token']
    content = "<h2>Media Search</h2>"
    if not access_token:
        return 'Missing Access Token'
    try:
        api = client.InstagramAPI(access_token=access_token)
        content += str(dir(api.media_search))
        media_search = api.media_search(q="cat", count = 10, lat="40.727184",lng="-73.995833",distance=5000)
        photos = []
        for media in media_search:
            photos.append('<img src="%s"/>' % media.get_standard_resolution_url())
        content += ''.join(photos)
        content += str(len(media_search))
        content += "<br />"
        content += str(media_search)
        content += "<br />"
        content += str(dir(media_search[-1]))
    except Exception as e:
        print(e)              
    return "%s %s <br/>Remaining API Calls = %s/%s" % (get_nav(),content,api.x_ratelimit_remaining,api.x_ratelimit)

def LabelFaces(imgURL):
    """
    TODO define label as color with a box
    Labels the face in the color image and returns
    the face region of insterest in grayscale and color
    
    Vars: 
        TODO document this
    """
    img = cv2.imread(imgURL)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img)
    gray_faces_rois = []
    color_faces_rois = []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        gray_faces_rois.append(gray[y:y+h, x:x+w])
        color_faces_rois.append(img[y:y+h, x:x+w])
    
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    return gray_faces_rois, color_faces_rois

bottle.run(app=app, host='localhost', port=8515, reloader=True)