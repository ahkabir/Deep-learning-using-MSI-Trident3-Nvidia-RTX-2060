from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from stream_out import StreamOut
from camera_in import *
from video_in import *
from pic_out import *
import numpy as np
import argparse
import os
import sys
import math
import cv2
import pdb
import random
import string

"""PED_DETECT Pedestrian detection implementation.

This describes:
  - methods needed to perform pedestrian detection.
  - it's taken from opencv and customized
"""

class PedestrianDetection(StreamOut):
    """
    This class contains all information needed to perform
    pedestrian detection.
    """
    video_in = 0

    def __init__(self, win_stride, padding, scale, meanshift, video_in=None,
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
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
        self.win_stride = eval(win_stride)
        self.padding = eval(padding)
        self.scale = scale
        self.meanShift = True if meanshift > 0 else False
        self.count = ''

    def inside(self, r, q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


    def draw_detections(self, frame, rects, thickness = 1):
        for x, y, w, h in rects:
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.15*w), int(0.05*h)
            cv2.rectangle(frame, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

    def put_box(self, frame):
        cv2.rectangle(frame, (10, 10), (200, 30), (255, 255, 255), -1)
        cv2.putText(frame, self.count, (10, 22), cv2.FONT_HERSHEY_PLAIN, 1.0, (30, 144, 255), 2);

    def ped_detection(self):
        """
        Performs all necessary work to do pedestrian detection.

        Returns
        ----------
        ret : True or False
        frame : processed frame with prediction
        """
        if self.video_in != None:
            ret, frame = self.video_in.get_a_frame()
        elif self.camera_in != None:
            ret, frame = self.camera_in.get_a_frame()
        if ret == True:
            #DEFAULT: hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
            found, w = self.hog.detectMultiScale(
                        frame,
                        winStride=self.win_stride,
                        padding=self.padding,
                        scale=self.scale,
                        useMeanshiftGrouping=self.meanShift
                        )
            found_filtered = []
            for ri, r in enumerate(found):
                for qi, q in enumerate(found):
                    if ri != qi and self.inside(r, q):
                        break
                    else:
                        found_filtered.append(r)
            self.draw_detections(frame, found)
            self.draw_detections(frame, found_filtered, 3)
            self.count = "number of people = %d" % (len(found))
            self.put_box(frame)
            return ret, frame
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
            ret, frame = self.ped_detection()
            if ret == True:
                yield cv2.imencode('.jpg', frame)[1].tobytes()
            else:
                break

    def run(self):
        """
        Application invokes this method to perform pedestrian detection.
        """
        while True:
            ret, frame = self.ped_detection()
            if ret == True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # output the recognized body
                if self.video_out != None:
                    self.video_out.display(frame)
                if self.pic_out != None:
                    self.pic_out.save_frame(frame)
        if self.video_out != None:
            cv2.destroyAllWindows()
