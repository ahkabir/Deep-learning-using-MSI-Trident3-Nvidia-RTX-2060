
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
#import requests

#class CameraIn():
#   def __init__(self):
#        print('CameraIn constructor')

class CameraAxis:
    def __init__(self, camera_ip):
        print('Axis constructor')
        # TODO: user please provide your camera URL here
        # TODO: construct the string using your camera username, password, url
        # TODO: and assign it to self.camera_url below
        # string = 'http://<username>:<password>@%s/mjpg/1/video.mjpg' % (camera_ip)
        self.camera_url = 'string' % (camera_ip)
        self.video_capture = cv2.VideoCapture(0)

    def get_a_frame(self):
        ret, frame = self.video_capture.read()
        if ret == True:
            return ret, frame
        elif ret == False:
            return False, None

class CameraAmcrest:
    def __init__(self, camera_ip):
        print('Amcrest constructor')
        # TODO: user please provide your camera URL here
        # TODO: construct the string using your camera username, password, url
        # TODO: and assign it to self.camera_url below
        # string = 'http://<username>:<password>@%s/mjpg/1/video.mjpg' % (camera_ip)
        self.camera_url = "string" % (camera_ip)
        self.video_capture = cv2.VideoCapture(self.camera_url)

    def get_a_frame(self):
        ret, frame = self.video_capture.read()
        if ret == True:
            return ret, frame
        elif ret == False:
            return False, None
