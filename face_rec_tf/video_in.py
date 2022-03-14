
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
class VideoIn():
    def __init__(self, video_in):
        """
        The constructor method video input feed class.

        Parameters
        ----------
        video_in : video file
            The video file from input feed.
        """
        self.video_in = video_in

        # Handle for getting frame from video.
        self.video_capture = cv2.VideoCapture(video_in)

    def get_a_frame(self):
        """
        The method grabs a frame from input video feed.

        Returns
        ----------
        ret : True or False
            Caller can use this boolean to check frame validity.
        frame : frame handle
            The frame handle that may or may not contain a valid frame.
        """
        ret, frame = self.video_capture.read()
        return ret, frame
