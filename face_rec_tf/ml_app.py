
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

def gen(ca):
    """Video streaming generator function."""
    while True:
        #print('in app gen get_frame')
        frame = ca.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)
        #time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    object_need_stream.start_thread()
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(object_need_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def ped_detection(args):
    global object_need_stream
    if args.camera:
        camera_ip = args.camera[0]
        camera_make = args.camera[1]
        if camera_make == 'AXIS':
            print('Axis chosen')
        elif camera_make == 'AMCREST':
            print('Amcrest chosen')
        else:
            print('Err: camera make should be AXIS or AMCREST')
            exit(0)

        if args.out:
            if args.out[0] == "video":
                if camera_make == 'AXIS':
                    p = PedestrianDetection('(8,8)', '(32,32)', 1.05, -1,
                                           None,
                                           CameraAxis(camera_ip),
                                           VideoOut('ped'),
                                           None)
                elif camera_make == 'AMCREST':
                    p = PedestrianDetection('(8,8)', '(32,32)', 1.05, -1,
                                           None,
                                           CameraAmcrest(camera_ip),
                                           VideoOut('ped'),
                                           None)
            elif args.out[0] == "stream":
                if camera_make == 'AXIS':
                    p = PedestrianDetection('(8,8)', '(32,32)', 1.05, -1,
                                           None,
                                           CameraAxis(camera_ip),
                                           None,
                                           None)
                elif camera_make == 'AMCREST':
                    p = PedestrianDetection('(8,8)', '(32,32)', 1.05, -1,
                                           None,
                                           CameraAmcrest(camera_ip),
                                           None,
                                           None)
            elif args.out[0] == "pic":
                tmp = 'recog_ped_camera_' + ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
                if camera_make == 'AXIS':
                    p = PedestrianDetection('(8,8)', '(32,32)', 1.05, -1,
                                           None,
                                           CameraAxis(camera_ip),
                                           None,
                                           PicOut(tmp))
                elif camera_make == 'AMCREST':
                    p = PedestrianDetection('(8,8)', '(32,32)', 1.05, -1,
                                           None,
                                           CameraAmcrest(camera_ip),
                                           None,
                                           PicOut(tmp))

    if args.video:
        PedestrianDetection.video_in = args.video[0]
        if args.out:
            if args.out[0] == "video":
                p = PedestrianDetection('(8,8)', '(32,32)', 1.05, -1,
                                       VideoIn(PedestrianDetection.video_in),
                                       None,
                                       VideoOut('ped'),
                                       None)

            elif args.out[0] == "stream":
                p = PedestrianDetection('(8,8)', '(32,32)', 1.05, -1,
                                       VideoIn(PedestrianDetection.video_in),
                                       None,
                                       None,
                                       None)
            elif args.out[0] == "pic":
                tmp = 'recog_ped_video_' + ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
                p = PedestrianDetection('(8,8)', '(32,32)', 1.05, -1,
                                       VideoIn(PedestrianDetection.video_in),
                                       None,
                                       None,
                                       PicOut(tmp))
    object_need_stream = p
    if args.body:
        if args.out[0] == "stream":
            #app.run(host='0.0.0.0', threaded=True)
            app.run(host='0.0.0.0', debug=True)
        else:
            p.run()


def training(args):
    FaceModel.mfile = 'models/20170512-110547.pb'
    FaceModel.cfile = 'models/face_demo_classifier.pkl'
    FaceModel.data_dir = 'datasets/my_dataset/train/'
    if args.camera:
        camera_ip = args.camera[0]
        camera_make = args.camera[1]
        if camera_make == 'AXIS':
            print('Axis chosen')
        elif camera_make == 'AMCREST':
            print('Amcrest chosen')
        else:
            print('Err: camera make should be AXIS or AMCREST')
            exit(0)

        if args.out:
            if args.out[0] == "video":
                if camera_make == 'AXIS':
                    t = FaceTraining(args.train[0], None, CameraAxis(camera_ip), VideoOut('face'), None)
                elif camera_make == 'AMCREST':
                    t = FaceTraining(args.train[0], None, CameraAmcrest(camera_ip), VideoOut('face'), None)
            elif args.out[0] == "pic":
                tmp = 'recog_train_camera_' + ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
                if camera_make == 'AXIS':
                    t = FaceTraining(args.train[0], None, CameraAxis(camera_ip), None, PicOut(tmp))
                elif camera_make == 'AMCREST':
                    t = FaceTraining(args.train[0], None, CameraAmcrest(camera_ip), None, PicOut(tmp))
            elif args.out[0] == "stream":
                print('Err: Output streaming not supported for training,choose [video|pic]')
                exit(0)
    if args.video:
        FaceModel.video_in = args.video[0]
        if args.out:
            if args.out[0] == "video":
                t = FaceTraining(args.train[0], VideoIn(FaceModel.video_in), None, VideoOut('face'), None)
            elif args.out[0] == "pic":
                tmp = 'recog_train_video_' + ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
                t = FaceTraining(args.train[0], VideoIn(FaceModel.video_in), None, None, PicOut(tmp))
            elif args.out[0] == "stream":
                print('Err: Output streaming not supported for training,choose [video|pic]')
                exit(0)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            print('Loading feature extraction model')
            facenet.load_model(t.mfile)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename_exp = os.path.expanduser(t.cfile)

            print('######################################################################')
            print('                             TRAINING STARTED                         ')
            print('######################################################################')

            # Save pictures for training
            t.run()

            # Now train
            dataset = facenet.get_dataset(FaceModel.data_dir)
            # FIXME : the assertion syntax caused python warning
            #for cls in dataset:
            #    assert(len(cls.image_paths)>0,
            #    'There must be at least one image for each class in the dataset')
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print('    Number of classes: %d' % len(dataset))
            print('    Number of images: %d' % len(paths))
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / t.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*t.batch_size
                end_index = min((i+1)*t.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, t.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            print('    Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)
            class_names = [ cls.name.replace('_', ' ') for cls in dataset]
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('    Saved classifier model to file "%s"' % classifier_filename_exp)


def classify(args):
    global object_need_stream
    FaceModel.mfile = 'models/20170512-110547.pb'
    FaceModel.cfile = 'models/face_demo_classifier.pkl'
    tmp = ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
    FaceModel.data_dir = 'delete_' + tmp

    if args.camera:
        camera_ip = args.camera[0]
        camera_make = args.camera[1]
        if camera_make == 'AXIS':
            print('Axis chosen')
        elif camera_make == 'AMCREST':
            print('Amcrest chosen')
        else:
            print('Err: camera make should be AXIS or AMCREST')
            exit(0)

        if args.out:
            if args.out[0] == "video":
                if camera_make == 'AXIS':
                    c = FaceClassification(None, CameraAxis(camera_ip), VideoOut('face'), None)
                elif camera_make == 'AMCREST':
                    c = FaceClassification(None, CameraAmcrest(camera_ip), VideoOut('face'), None)
            elif args.out[0] == "stream":
                if camera_make == 'AXIS':
                    c = FaceClassification(None, CameraAxis(camera_ip),None, None)
                elif camera_make == 'AMCREST':
                    c = FaceClassification(None, CameraAmcrest(camera_ip),None, None)
            elif args.out[0] == "pic":
                tmp = 'recog_classify_camera_' + ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
                if camera_make == 'AXIS':
                    c = FaceClassification(None, CameraAxis(camera_ip),None, PicOut(tmp))
                elif camera_make == 'AMCREST':
                    c = FaceClassification(None, CameraAmcrest(camera_ip),None, PicOut(tmp))

    if args.video:
        FaceModel.video_in = args.video[0]
        if args.out:
            if args.out[0] == "video":
                c = FaceClassification(VideoIn(FaceModel.video_in), None,
                                             VideoOut('face'),
                                             None)
            elif args.out[0] == "stream":
                c = FaceClassification(VideoIn(FaceModel.video_in), None,
                                             None,
                                             None)
            elif args.out[0] == "pic":
                tmp = 'recog_classify_video_' + ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
                c = FaceClassification(VideoIn(FaceModel.video_in), None,
                                             None,
                                             PicOut(tmp))
    object_need_stream = c

    with tf.Graph().as_default():
            with tf.Session() as sess:
                print('Loading feature extraction model')
                facenet.load_model(c.mfile)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                classifier_filename_exp = os.path.expanduser(c.cfile)

                print('######################################################################')
                print('                         CLASSIFICATION STARTED                       ')
                print('######################################################################')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                print('    Loaded classifier model from file "%s"' % classifier_filename_exp)

                FaceModel.images_placeholder = images_placeholder
                FaceModel.phase_train_placeholder = phase_train_placeholder
                FaceModel.embeddings = embeddings
                FaceModel.embedding_size = embedding_size
                FaceModel.sess = sess
                FaceModel.model = model
                FaceModel.class_names = class_names
                if args.classify:
                    if args.out[0] == "stream":
                        app.run(host='0.0.0.0', threaded=True)
                    else:
                        c.run()


def obj_detection(args):
    global object_need_stream
    tmp = ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
    if args.camera:
        camera_ip = args.camera[0]
        camera_make = args.camera[1]
        if camera_make == 'AXIS':
            print('Axis chosen')
        elif camera_make == 'AMCREST':
            print('Amcrest chosen')
        else:
            print('Err: camera make should be AXIS or AMCREST')
            exit(0)

        if args.out:
            if args.out[0] == "video":
                if camera_make == 'AXIS':
                    o = ObjectDetection()
                elif camera_make == 'AMCREST':
                    o = ObjectDetection(None, CameraAmcrest(camera_ip), VideoOut('obj'), None)
            elif args.out[0] == "stream":
                if camera_make == 'AXIS':
                    o = ObjectDetection(None, CameraAxis(camera_ip),None, None)
                elif camera_make == 'AMCREST':
                    o = ObjectDetection(None, CameraAmcrest(camera_ip),None, None)
            elif args.out[0] == "pic":
                tmp = 'recog_object_camera_' + ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
                if camera_make == 'AXIS':
                    o = ObjectDetection(None, CameraAxis(camera_ip),None, PicOut(tmp))
                elif camera_make == 'AMCREST':
                    o = ObjectDetection(None, CameraAmcrest(camera_ip),None, PicOut(tmp))

    if args.video:
        if args.out:
            if args.out[0] == "video":
                o = ObjectDetection(VideoIn(args.video[0]), None,
                                             VideoOut('obj'),
                                             None)
            elif args.out[0] == "stream":
                o = ObjectDetection(VideoIn(args.video[0]), None,
                                             None,
                                             None)
            elif args.out[0] == "pic":
                tmp = 'recog_object_video_' + ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
                o = ObjectDetection(VideoIn(args.video[0]), None,
                                             None,
                                             PicOut(tmp))
    object_need_stream = o
    if args.object:
        if args.out[0] == "stream":
            app.run(host='0.0.0.0', threaded=True)
        else:
            o.run()


def main(args):
    if args.classify:
        global c
        classify(args)
    if args.train:
        training(args)
    if args.body:
        ped_detection(args)
    if args.object:
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
