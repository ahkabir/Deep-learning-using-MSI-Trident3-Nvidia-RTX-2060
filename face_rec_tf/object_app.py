
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from face import *
from face_cl import *
from face_tr import *
from ped_detect import *
from camera_in import *
from video_in import *
from pic_out import *
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
import cv2
import pdb
import shutil
import requests
import time
import datetime
import random
import string
from sklearn.svm import SVC
from sklearn.svm import SVC
from object_detection.obj_detect import *

from flask import Flask, render_template, Response

object_need_stream = None

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen(vs):
    """Video streaming generator function."""
    while True:
        #print('in app gen get_frame')
        frame = vs.readDetections()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)
        #time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    object_need_stream.start()
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(object_need_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def obj_detection(args):
    print('AK get object')
    o = ObjectDetection()
    global object_need_stream
    object_need_stream = o
    if args.object:
        if args.out[0] == "stream":
            print('AK start stream')
            app.run(host='0.0.0.0', threaded=True)


def main(args):
    if args.classify:
        pass
    if args.train:
        pass
    if args.body:
        pass
    if args.object:
        print('AK object detection')
        obj_detection(args)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # app grouping
    app = parser.add_mutually_exclusive_group(required=True)
    app.add_argument('--body',
          help='Run body detection app', action="store_true")
    app.add_argument('--object',
          help='Run object detection app', action="store_true")
    app.add_argument('--classify',
          help='Run face recoginition', action="store_true")
    app.add_argument('--train', nargs=1, metavar=('name'),
          help='Run training for face recognition for name')

    # input grouping
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--camera', nargs=2, metavar=('ip', 'make'),
          help='The IP address and make = [AXIS|AMCREST]')
    group.add_argument('--video', nargs=1, metavar=('file'),
          help='The classification <file> : video or still')

    # output
    parser.add_argument('--out', required=True, nargs=1, metavar=('sink'),
          help='sink could be [\"stream\" | \"video\" | \"pic\"]')
    return parser.parse_args(argv)

if __name__ == '__main__':
    #print(sys.argv[1:])
    main(parse_arguments(sys.argv[1:]))
