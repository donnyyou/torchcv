import os
import math
import shutil
import cv2
import numpy as np
from skimage import io

# https://github.com/1adrianb/face-alignment pip install face_alignment
import face_alignment as fa

from tools.helper.file_helper import FileHelper
from tools.helper.image_helper import ImageHelper
from tools.util.logger import Logger as Log


class FaceAlignmentor(object):

    def __init__(self, dist_ec_mc, ec_y, crop_size=144):
        self.dist_ec_mc = dist_ec_mc
        self.ec_y = ec_y
        self.crop_size = 144
        self.face_detector = fa.FaceAlignment(fa.LandmarksType._2D, device='cuda', flip_input=False)

    def detect_face(self, img):
        preds = self.face_detector.get_landmarks(img)
        if preds is None or len(preds) != 1:
            return None

        f68pt = preds[0]
        out = np.array([(f68pt[36] + f68pt[39]) / 2, (f68pt[42] + f68pt[45]) / 2, f68pt[33], f68pt[48], f68pt[54]])
        return out

    def align_face(self, img, f5pt):
        ang_tan = (f5pt[0,1] - f5pt[1, 1]) / (f5pt[0, 0]-f5pt[1, 0])
        rotate_degree = math.atan(ang_tan) / math.pi * 180
        height, width, _ = img[0].shape if isinstance(img, (list, tuple)) else img.shape

        img_center = (width / 2.0, height / 2.0)

        rotate_mat = cv2.getRotationMatrix2D(img_center, rotate_degree, 1.0)
        cos_val = np.abs(rotate_mat[0, 0])
        sin_val = np.abs(rotate_mat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotate_mat[0, 2] += (new_width / 2.) - img_center[0]
        rotate_mat[1, 2] += (new_height / 2.) - img_center[1]
        if isinstance(img, (list, tuple)):
            for i in range(len(img)):
                img[i] = cv2.warpAffine(img[i], rotate_mat, (new_width, new_height), borderValue=0).astype(np.uint8)
        else:
            img = cv2.warpAffine(img, rotate_mat, (new_width, new_height), borderValue=0).astype(np.uint8)

        for i in range(len(f5pt)):
            x = f5pt[i][0]
            y = f5pt[i][1]
            p = np.array([x, y, 1])
            p = rotate_mat.dot(p)
            f5pt[i][0] = p[0]
            f5pt[i][1] = p[1]

        r_scale = self.dist_ec_mc / ((f5pt[3, 1] + f5pt[4, 1]) / 2 - (f5pt[0, 1] + f5pt[1, 1]) / 2)
        height, width, _ = img[0].shape if isinstance(img, (list, tuple)) else img.shape
        target_size = [int(width * r_scale), int(height * r_scale)]
        if r_scale < 0:
            return None, None

        if isinstance(img, (list, tuple)):
            for i in range(len(img)):
                img[i] = ImageHelper.resize(img[i], target_size, interpolation='cubic')
        else:
            img = ImageHelper.resize(img, target_size, interpolation='cubic')
        f5pt = f5pt * r_scale

        crop_y = max(int((f5pt[0, 1] + f5pt[1, 1]) / 2 - self.ec_y), 0)
        crop_x = max(int((f5pt[0, 0] + f5pt[1, 0]) / 2 - self.crop_size // 2), 0)
        f5pt[:, 0] -= crop_x
        f5pt[:, 1] -= crop_y
        if isinstance(img, (list, tuple)):
            for i in range(len(img)):
                img[i] = img[i][crop_y:crop_y+self.crop_size, crop_x:crop_x+self.crop_size]
        else:
            img = img[crop_y:crop_y+self.crop_size, crop_x:crop_x+self.crop_size]
        return img, f5pt

    def process(self, data_dir):
        new_data_dir = '{}_new'.format(data_dir.rstrip('/'))
        if os.path.exists(new_data_dir):
            shutil.rmtree(new_data_dir)

        os.makedirs(new_data_dir)

        for filename in FileHelper.list_dir(data_dir):
            if not ImageHelper.is_img(filename):
                Log.info('Image Path: {}'.format(os.path.join(data_dir, filename)))
                continue

            file_path = os.path.join(data_dir, filename)
            img = io.imread(file_path)
            kpts = self.detect_face(img)
            if kpts is None:
                Log.info('Invliad face detected in {}'.format(file_path))
                continue

            face, kpts = self.align_face(img, kpts)
            cv2.imwrite(os.path.join(new_data_dir, filename), ImageHelper.rgb2bgr(face))

    def process_3d(self, data_dir):
        new_data_dir = '{}_new'.format(data_dir.rstrip('/'))
        if os.path.exists(new_data_dir):
            shutil.rmtree(new_data_dir)

        os.makedirs(new_data_dir)

        for filename in FileHelper.list_dir(data_dir):
            if not ImageHelper.is_img(filename) or 'depth' in filename:
                Log.info('Image Path: {}'.format(os.path.join(data_dir, filename)))
                continue

            file_path = os.path.join(data_dir, filename)
            img = io.imread(file_path)
            kpts = self.detect_face(img)
            if kpts is None:
                Log.info('Invliad face detected in {}'.format(file_path))
                continue

            depth = np.array(io.imread(os.path.join(data_dir, filename.replace('rgb', 'depth'))))
            face_depth, kpts = self.align_face([np.array(img), np.array(depth)], kpts)
            if face_depth is None:
                Log.info('Invliad face detected in {}'.format(file_path))
                continue
            ImageHelper.save(ImageHelper.rgb2bgr(face_depth[0]), os.path.join(new_data_dir, filename))
            ImageHelper.save(ImageHelper.rgb2bgr(face_depth[1]), os.path.join(new_data_dir, filename.replace('rgb', 'depth')))


if __name__ == '__main__':
    face_alignmentor = FaceAlignmentor(48, 48)
    face_alignmentor.process_3d('/home/donny/DataSet/GAN/3D2VIS')
