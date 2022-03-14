from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""FACE_CL Contains face recognition classification related data.

This describes:
  - methods needed to perform face classification.
"""

import cv2
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


class FaceClassification(FaceModel, StreamOut):
    """
    This class contains all information needed to perform
    face recognition classification.
    """
    def __init__(self, video_in=None,
                camera_in=None, video_out=None,
                pic_out=None):
        StreamOut.__init__(self)
        """
        The constructor method for face classification.

        Parameters
        ----------
        video_in : Class VideoIn
            Non null means the input feed is video.
        camera_in : Class CameraIn
            Non null means the input feed is camera.
        video_out : VideoOut
            Non null means the output feed is video.
        pic_out : PicOut
            Non null means the output feed is file(s).
        """
        # composition : i/p interface
        self.video_in = video_in
        self.camera_in = camera_in
        # composition : o/p interface
        self.video_out = video_out
        self.pic_out = pic_out

    def face_prediction(self, frame, faces):
        """
        Predicts whether the faces belong to a trained class.

        Parameters
        ----------
        frame : numpy array
            Original frame captured from camera or video.
        faces : numpy array.
            The detected face. Caller will ensure that faces is not empty.

        Returns
        ----------
        frame : recognized frame with bounding box and name(identified/unknown)
        """
        predictions = FaceModel.model.predict_proba(FaceModel.emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[
                                        np.arange(len(best_class_indices)),
                                        best_class_indices
                                              ]
        print('    prediction:')
        rec_name_lst = []
        for i in range(len(best_class_indices)):
            print('    %4d  %s: %.3f' % (
                                  i,
                                  FaceModel.class_names[best_class_indices[i]],
                                  best_class_probabilities[i]
                                        )
                 )
            accuracy = np.mean(np.equal(best_class_indices, FaceModel.labels))
            rec_name = FaceModel.class_names[best_class_indices[i]]
            if best_class_probabilities[i] < 0.7:
                rec_name = "unknown"
            rec_name_lst.append(rec_name)
        print('    Accuracy: %.3f' % accuracy)
        j = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x-20, y-20),
                         (x+w +20, y+h+20), (0, 255, 0), 4)
            cv2.putText(frame, rec_name_lst[j], (x, y),
                       cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2);
            j = j + 1
        return frame

    def classification(self):
        """
        Performs all necessary work to do face classification.

        Returns
        ----------
        boolean : True or False; True indicates frame is non empty
        frame : non-empty frame or frame recognized or empty frame
        """
        if self.video_in != None:
            ret, frame = self.video_in.get_a_frame()
        elif self.camera_in != None:
            ret, frame = self.camera_in.get_a_frame()
        if ret == True:
            # detect face
            faces = FaceModel.detect_face(self, frame)
            FaceModel.write_faces_to_file(self, frame, faces)
            status = FaceModel.face_embeddings(self, faces)
            if status == True:
                bounded_frame = self.face_prediction(frame, faces)
                # We are done with embedding and prediction.
                # We can delete the temp directory where we saved
                # the frame, so that the next frame with face
                # can be saved there
                shutil.rmtree(FaceModel.data_dir)
                os.makedirs(FaceModel.data_dir)
                return True, bounded_frame
            else:
                return True, frame
        else:
            return False, None

    def frames(self):
        """
        This method is called by the StreamOut parent class.
        The StreamOut class has a thread that periodically
        calls this method to grab a frame and provide that to
        web server.
        """
        while True:
            ret, frame = self.classification()
            if ret == True:
                yield cv2.imencode('.jpg', frame)[1].tobytes()
            else:
                break

    def run(self):
        """
        ML application invokes this method to perform classification.
        """
        while True:
            ret, frame = self.classification()
            # valid frame
            if ret == True:
                # output the recognized face
                if self.video_out != None:
                    self.video_out.display(frame)
                if self.pic_out != None:
                    self.pic_out.save_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if self.video_out != None:
            cv2.destroyAllWindows()

