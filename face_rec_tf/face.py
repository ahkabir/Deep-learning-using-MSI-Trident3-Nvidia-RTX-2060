"""FACE Parent class definition for face recognition.

This describes:
  - common methods to be used both in training and classification.
  - user provided configuration data that is used by training and classification.

Prgoram that need to perform face recognition should import this
file. The training and classification classes can be sub classes
of this class.
"""
import cv2
import os
import facenet
import math
import numpy as np

class FaceModel:
    """
    This class contains all common information needed to
    perform face recognition training and classification.
    """
    # common
    mfile = 0
    cfile = 0
    batch_size = 1000
    image_size = 160
    video_in = 0
    video_out = 0
    data_dir = 0
    faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    # classification
    embedding_size = 0
    images_placeholder = 0
    phase_train_placeholder = 0
    embeddings = 0
    sess = 0
    model = 0
    class_names = 0
    emb_array = 0
    labels = 0

    # training
    person = 0

    def detect_face(self, frame):
        """
        This method detects a face if any given a frame from an image.

        Parameters
        ----------
        frame : numpy array
            The extracted image frame from camera or file.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        return faces

    def write_faces_to_file(self, frame, faces):
        """
        This method writes each face of a given frame to a file.
        The given frame may contain one or more faces.
        The writeen file is resized to dimension 180x200

        Parameters
        ----------
        frame : numpy array
            The extracted image frame from camera or file.

        faces : detected by haar cascade.
        """
        if len(faces):
            i = 0
            for (x, y, w, h) in faces:
                sub_face = frame[y:y+h, x:x+w]
                i = i + 1
                DirName = (FaceModel.data_dir + ("/dir%d/" % (i)))
                if not os.path.exists(DirName):
                    os.makedirs(DirName)
                FaceFileName = DirName + ("face%d.jpg" % (i))
                resized_image = cv2.resize(sub_face, (180, 200))
                cv2.imwrite(FaceFileName, resized_image)
            return True
        else:
            return False

    def face_embeddings(self, faces):
        """
        This method performs the tensorflow embeddings on a given face.

        Parameters
        ----------
        faces : detected by haar cascade.
        """
        if len(faces):
            dataset = facenet.get_dataset(FaceModel.data_dir)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print('    Calculating features for images : %d' % len(faces))
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / FaceModel.batch_size))
            emb_array = np.zeros((nrof_images, FaceModel.embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*FaceModel.batch_size
                end_index = min((i+1)*FaceModel.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, FaceModel.image_size)
                feed_dict = { FaceModel.images_placeholder:images, FaceModel.phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = FaceModel.sess.run(FaceModel.embeddings, feed_dict=feed_dict)
            FaceModel.emb_array = emb_array
            FaceModel.labels = labels
            return True
        else:
            return False
