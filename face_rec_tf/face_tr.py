from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from stream_out import StreamOut
from face import *
from camera_in import *
from video_out import *
from pic_out import *
import numpy as np
import argparse
import facenet
import os
import sys
import math
import cv2
import shutil
from sklearn.svm import SVC
from sklearn.svm import SVC

class FaceTraining(FaceModel):
    """
    This class contains all information needed to perform
    face recognition training.
    """
    def __init__(self, person, video_in=None,
                camera_in=None, video_out=None,
                pic_out=None):
        """
        The constructor method for face training.

        Parameters
        ----------
        person : string
            Person to be trained.
        video_in : Class VideoIn
            Non null means the input feed is video.
        camera_in : Class CameraIn
            Non null means the input feed is camera.
        video_out : VideoOut
            Non null means the output feed is video.
        pic_out : PicOut
            Non null means the output feed is file(s).
        """
        self.person = person

        # composition : i/p interface
        self.video_in = video_in
        self.camera_in = camera_in

        # composition : o/p interface
        self.video_out = video_out
        self.pic_out = pic_out
        self.index = 0

        # following co-ordinates for creating vertical
        # box to show training status. 
        self.x1 = 15
        self.x2 = 35
        self.y1 = 340
        self.y2 = 340
        self.train_dir_name = ''

    def save_classification_model_file(self):
        """
        This method saves the training file.
        """
        classifier_filename_exp = os.path.expanduser(self.cfile)
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('    Saved classifier model to file "%s"' % classifier_filename_exp)

    def save_training_pics(self, frame, faces):
        """
        During training 100 pictures are take of each person.
        These pictures need to be saved to perform training among
        all the other pictures in the datasets/ sub directory.
        Note that the face dimension is reduced to 180x200 and
        then saved.

        Parameters
        ----------
        frame : numpy array
            Original frame captured from camera or video.
        faces : numpy array.
            The detected face.
        """
        for (x, y, w, h) in faces:
            sub_face = frame[y:y+h, x:x+w]
            self.index = self.index + 1
            FaceFileName = self.train_dir_name + ("face%d.jpg" % (self.index))
            resized_image = cv2.resize(sub_face, (180, 200))
            cv2.imwrite(FaceFileName, resized_image)

    def put_box(self, frame, faces):
        """
        This method puts box around the detected face(s).
        """
        # outer empty box to indicate full training time
        # white : (255,255,255); filled because of -1
        cv2.rectangle(frame, (10, 340), (40, 40), (255, 255, 255), -1)
        if len(faces):
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x-20, y-20), (x+w +20, y+h+20), (0, 255, 0), 4)
                banner = "Training: %s fr %d" % (self.person, self.index)
                cv2.putText(frame, banner, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2);
            self.y2 = self.y2 - 3
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (160, 160, 160), -1)
        return frame

    def training_data_save(self):
        """
        This method performs all work needed for training.
        It gets a frame from input feed, detects it, put
        a box around it and pass the frame handle with status
        to caller.
        """
        if self.video_in != None:
            ret, frame = self.video_in.get_a_frame()
        elif self.camera_in != None:
            ret, frame = self.camera_in.get_a_frame()
        if ret == True:
            # detect face
            faces = FaceModel.detect_face(self, frame)
            # There may or may not be any faces in frame
            # Do embeddings only when faces are found
            if len(faces):
                self.save_training_pics(frame, faces)
                frame = self.put_box(frame, faces)
            return ret, frame
        else:
            return False, None

    def run(self):
        """
        This method is called by the training application.
        """
        self.train_dir_name = FaceModel.data_dir + self.person + '/'
        if not os.path.exists(self.train_dir_name):
            os.makedirs(self.train_dir_name)
        while True:
            ret, frame = self.training_data_save()
            if ret == True:
                if self.video_out != None:
                    self.video_out.display(frame)
                if self.pic_out != None:
                    self.pic_out.save_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if self.index == 100:
                print('Captured 100 frames of %s (done)' % (self.person))
                break
        if self.video_out != None:
            cv2.destroyAllWindows()
