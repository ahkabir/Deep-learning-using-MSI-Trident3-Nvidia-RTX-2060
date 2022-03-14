
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from face import *
import cv2
import shutil

class PicOut():
    def __init__(self, dir):
        self.dir = dir
        self.count = 0
        if not os.path.exists(dir):
            os.makedirs(dir)
    def save_frame(self, frame):
        # write to file
        self.count = self.count + 1
        file_name = self.dir + ('/pic%d.jpg' % self.count)
        cv2.imwrite(file_name, frame)
        #wipe out the per face directory
        #shutil.rmtree(FaceModel.data_dir)
        #os.makedirs(FaceModel.data_dir)
