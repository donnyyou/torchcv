#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# COCO det data generator.


import json
import os
import argparse
import shutil

from pycocotools.coco import COCO


JOSN_DIR = 'json'
IMAGE_DIR = 'image'

CAT_DICT = {
    "person": 1, "bicycle":2, "car": 3, "motorcycle": 4, "airplane": 5, "bus": 6, "train": 7,
    "truck": 8, "boat": 9, "traffic light": 10, "fire hydrant": 11, "stop sign": 13, "parking meter": 14,
    "bench": 15, "bird": 16, "cat": 17, "dog": 18, "horse": 19, "sheep": 20, "cow": 21, "elephant": 22,
    "bear": 23, "zebra": 24, "giraffe": 25, "backpack": 27, "umbrella": 28, "handbag": 31, "tie": 32,
    "suitcase": 33, "frisbee": 34, "skis": 35, "snowboard": 36, "sports ball": 37, "kite": 38,
    "baseball bat": 39, "baseball glove": 40, "skateboard": 41, "surfboard": 42, "tennis racket": 43,
    "bottle": 44, "wine glass": 46, "cup": 47, "fork": 48, "knife": 49, "spoon": 50, "bowl": 51,
    "banana": 52, "apple": 53, "sandwich": 54, "orange": 55, "broccoli": 56, "carrot": 57, "hot dog": 58,
    "pizza": 59, "donut": 60, "cake": 61, "chair": 62, "couch": 63, "potted plant": 64, "bed": 65,
    "dining table": 67, "toilet": 70, "tv": 72, "laptop": 73, "mouse": 74, "remote": 75, "keyboard": 76,
    "cell phone": 77, "microwave": 78, "oven": 79, "toaster": 80, "sink": 81, "refrigerator": 82,
    "book": 84, "clock": 85, "vase": 86, "scissors": 87, "teddy bear": 88, "hair drier": 89, "toothbrush": 90
}


class CocoDetGenerator(object):

    def __init__(self, args, json_dir=JOSN_DIR, image_dir=IMAGE_DIR):
        self.args = args
        self.json_dir = os.path.join(self.args.save_dir, json_dir)
        if not os.path.exists(self.json_dir):
            os.makedirs(self.json_dir)

        self.image_dir = os.path.join(self.args.save_dir, image_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.coco = COCO(self.args.anno_file)
        self.cat_ids = self.coco.getCatIds(catNms=CAT_DICT.keys())
        print(len(self.cat_ids))
        self.img_ids = list(self.coco.imgs.keys())

    def generate_label(self):
        for i, img_id in enumerate(self.img_ids):
            json_dict = dict()
            file_name = self.coco.imgs[img_id]['file_name']
            json_dict['width'] = self.coco.imgs[img_id]['width']
            json_dict['height'] = self.coco.imgs[img_id]['height']

            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
            annos = self.coco.loadAnns(ann_ids)
            object_list = list()
            for anno in annos:
                object_dict = dict()
                object_dict['label'] = self.cat_ids.index(anno['category_id'])
                bbox = anno['bbox']
                object_dict['bbox'] = [float(bbox[0]), float(bbox[1]),
                                       (float(bbox[2]) + float(bbox[0])), (float(bbox[3]) + float(bbox[1]))]
                object_dict['segm'] = anno['segmentation']
                # Sanitize bboxes -- some are invalid
                if anno['area'] > 0 and float(bbox[2]) > 0 and float(bbox[3]) > 0:
                    object_list.append(object_dict)

            json_dict['objects'] = object_list
            fw = open(os.path.join(self.json_dir, '{}.json'.format(file_name.split('.')[0])), 'w')
            fw.write(json.dumps(json_dict))
            fw.close()
            shutil.copy(os.path.join(self.args.ori_img_dir, file_name),
                        os.path.join(self.image_dir, file_name))

            if i % 1000 == 0:
                print("Processed {} of {}".format(i, len(self.img_ids)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--ori_img_dir', default=None, type=str,
                        dest='ori_img_dir', help='The directory of the image data.')
    parser.add_argument('--anno_file', default=None, type=str,
                        dest='anno_file', help='The annotation file.')

    args = parser.parse_args()

    coco_det_generator = CocoDetGenerator(args)
    coco_det_generator.generate_label()
