
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from face import *
import cv2
import shutil

class VideoOut():
    def __init__(self, name):
        """
        The constructor method video output feed class.

        Parameters
        ----------
        name : string
            The name of the display window where the output video will
            be shown.
        """
        self.display_name = name

    def display(self, frame):
        """
        This method is called by classification processing.
        As part of classification frames are grabbed from input feed
        one frame at a time. For each of these frame, frame is saved,
        then displayed and then the frame is deleted so that the next
        new one could be brought in.
        """
        #shutil.rmtree(FaceModel.data_dir)
        #os.makedirs(FaceModel.data_dir)
        cv2.imshow(self.display_name, frame)

    def display_only(self, frame):
        """
        This method is called by training processing.
        It simply displays the frame to output video while a person
        is being trained standing in front of camera.
        """
        cv2.imshow(self.display_name, frame)
