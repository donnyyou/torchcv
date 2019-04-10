import os
import math
import shutil
import cv2
import numpy as np
from skimage import io

# https://github.com/1adrianb/face-alignment pip install face_alignment
import face_alignment as fa

from utils.helpers.file_helper import FileHelper
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log


class FaceAlignmentor(object):

    def __init__(self, dist_ec_mc, ec_y):
        self.dist_ec_mc = dist_ec_mc
        self.ec_y = ec_y
        self.face_detector = fa.FaceAlignment(fa.LandmarksType._2D, device='cpu', flip_input=False)

    def detect_face(self, img):
        preds = self.face_detector.get_landmarks(img)
        return preds

    def align_face(self, img, f5pt):
        ang_tan = (f5pt[0,1] - f5pt[1, 1]) / (f5pt[0, 0]-f5pt[1, 0])
        rotate_degree = math.atan(ang_tan) / math.pi * 180
        height, width, _ = img.shape

        img_center = (width / 2.0, height / 2.0)

        rotate_mat = cv2.getRotationMatrix2D(img_center, rotate_degree, 1.0)
        cos_val = np.abs(rotate_mat[0, 0])
        sin_val = np.abs(rotate_mat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotate_mat[0, 2] += (new_width / 2.) - img_center[0]
        rotate_mat[1, 2] += (new_height / 2.) - img_center[1]
        img = cv2.warpAffine(img, rotate_mat, (new_width, new_height), borderValue=0).astype(np.uint8)

        for i in range(len(f5pt)):
            x = f5pt[i][0]
            y = f5pt[i][1]
            p = np.array([x, y, 1])
            p = rotate_mat.dot(p)
            f5pt[i][0] = p[0]
            f5pt[i][1] = p[1]

        r_scale = self.dist_ec_mc / ((f5pt[3, 1] + f5pt[4, 1]) / 2 - (f5pt[0, 1] - f5pt[1, 1]) / 2)
        target_size = [int(i * r_scale) for i in img.shape[:2]]
        img = ImageHelper.resize(img, target_size[::-1], interpolation='cubic')
        f5pt = f5pt * r_scale
        return img, f5pt

    def process(self, data_dir):
        new_data_dir = '{}_new'.format(data_dir.rstrip('/'))
        if os.path.exists(new_data_dir):
            shutil.rmtree(new_data_dir)

        os.makedirs(new_data_dir)

        for filename in FileHelper.list_dir(FileHelper):
            if not ImageHelper.is_img(filename):
                continue

            file_path = os.path.join(data_dir, filename)
            img = io.imread(file_path)
            kpts = self.detect_face(img)
            if kpts is None:
                Log.info('No face detected in {}'.format(file_path))
            face, kpts = self.align_face(img, kpts)
            cv2.imshow('main', face)
            cv2.waitKey()


if __name__ == '__main__':
    face_alignmentor = FaceAlignmentor(48, 48)
    face_alignmentor.process('./test')
