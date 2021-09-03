import os
import numpy as np
import cv2 as cv
from datetime import datetime


class DatabaseHandler(object):
    """
    Handle the creation and saving of directories and images  for faces and audio
    """

    def __init__(self, root_path: str):
        self.root_dir = root_path

    def _check_dir_exist(self, name: str):
        return os.path.exists(os.path.join(self.root_dir, name))

    def _add_face(self, name: str, img_face: np.ndarray, emb_face: list):
        face_dir = os.path.join("face", name)
        if not self._check_dir_exist(face_dir):
            os.mkdir(os.path.join(self.root_dir, face_dir))

        img_filename = str(datetime.timestamp(datetime.now())).replace('.', '_')
        filename_img = os.path.join(self.root_dir, face_dir,  (img_filename + ".jpg"))
        filename_emb = os.path.join(self.root_dir, face_dir,  (img_filename + ".npy"))
        cv.imwrite(filename_img, img_face)
        np.save(filename_emb, emb_face)
        return True

    def save_faces(self, face_imgs: list, face_emb: list, name: str):
        name = name.lower()
        for face_img, face_emb in zip(face_imgs, face_emb):
            if not np.all((face_img == 0)):
                self._add_face(name, face_img, face_emb)

