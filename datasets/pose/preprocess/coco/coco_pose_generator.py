#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import os
import math
import json
import shutil
import argparse
import numpy as np
from PIL import Image

from pycocotools.coco import COCO


COCO_TO_OURS = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

JOSN_DIR = 'json'
MASK_DIR = 'mask'
IMAGE_DIR = 'image'


class CocoPoseGenerator(object):

    def __init__(self, args, json_dir=JOSN_DIR, mask_dir=MASK_DIR, image_dir=IMAGE_DIR):
        self.args = args
        self.json_dir = os.path.join(self.args.root_dir, json_dir)
        if not os.path.exists(self.json_dir):
            os.makedirs(self.json_dir)

        self.image_dir = os.path.join(self.args.root_dir, image_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.mask_dir = os.path.join(self.args.root_dir, mask_dir)
        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir)

        self.coco = COCO(self.args.anno_file)
        self.img_ids = list(self.coco.imgs.keys())

    def generate_label(self):
        for i, img_id in enumerate(self.img_ids):
            json_dict = dict()
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            img_anns = self.coco.loadAnns(ann_ids)
            num_persons = len(img_anns)
            filename = self.coco.imgs[img_id]['file_name']
            width = self.coco.imgs[img_id]['width']
            height = self.coco.imgs[img_id]['height']
            json_dict['height'] = height
            json_dict['width'] = width

            mask_list = list()

            persons = list()
            person_centers = list()

            for p in range(num_persons):
                if img_anns[p]['iscrowd'] == 1:
                    continue

                if img_anns[p]['num_keypoints'] < 5 or img_anns[p]['area'] < 32 * 32:
                    mask_list.append(p)
                    continue

                kpt = img_anns[p]['keypoints']
                dic = dict()

                # person center
                person_center = [img_anns[p]['bbox'][0] + img_anns[p]['bbox'][2] / 2.0,
                                 img_anns[p]['bbox'][1] + img_anns[p]['bbox'][3] / 2.0]

                # skip this person if the distance to exiting person is too small
                flag = 0
                for pc in person_centers:
                    dis = math.sqrt((person_center[0] - pc[0]) * (person_center[0] - pc[0])
                                    + (person_center[1] - pc[1]) * (person_center[1] - pc[1]))
                    if dis < pc[2] * 0.3:
                        flag = 1
                        break

                if flag == 1:
                    mask_list.append(p)
                    continue

                dic['bbox'] = [img_anns[p]['bbox'][0], img_anns[p]['bbox'][1],
                               img_anns[p]['bbox'][0] + img_anns[p]['bbox'][2],
                               img_anns[p]['bbox'][1] + img_anns[p]['bbox'][3]]

                dic['kpts'] = np.zeros((17, 3)).tolist()
                for part in range(17):
                    dic['kpts'][part][0] = kpt[part * 3]
                    dic['kpts'][part][1] = kpt[part * 3 + 1]
                    # visiable is 1, unvisiable is 0 and not labeled is -1
                    if kpt[part * 3 + 2] == 2:
                        dic['kpts'][part][2] = 1
                    elif kpt[part * 3 + 2] == 1:
                        dic['kpts'][part][2] = 0
                    else:
                        dic['kpts'][part][2] = -1

                persons.append(dic)
                person_centers.append(np.append(person_center, max(img_anns[p]['bbox'][2], img_anns[p]['bbox'][3])))

            if len(persons) > 0:
                persons = self.__coco_to_ours(persons)
                json_dict['objects'] = persons

                mask_all = np.zeros((height, width), dtype=np.uint8)
                mask_miss = np.zeros((height, width), dtype=np.uint8)
                flag = 0
                for p in range(num_persons):
                    if img_anns[p]['iscrowd'] == 1:
                        mask_crowd = self.coco.annToMask(img_anns[p])
                        temp = np.bitwise_and(mask_all, mask_crowd)
                        mask_crowd = mask_crowd - temp
                        flag += 1
                        continue
                    else:
                        mask = self.coco.annToMask(img_anns[p])

                    if p in mask_list:
                        mask_miss = np.bitwise_or(mask, mask_miss)
                    else:
                        mask_all = np.bitwise_or(mask, mask_all)

                if flag < 1:
                    mask_miss = np.logical_not(mask_miss)

                elif flag == 1:
                    mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
                else:
                    raise Exception('crowd segments > 1')

                fw = open(os.path.join(self.json_dir, '{}.json'.format(filename.split('.')[0])), 'w')
                fw.write(json.dumps(json_dict))
                fw.close()
                mask_miss_vis = np.zeros((height, width), dtype=np.uint8)
                mask_miss_vis[:, :] = mask_miss * 255
                mask_image = Image.fromarray(mask_miss, mode='P')
                mask_image.save(os.path.join(self.mask_dir, '{}.png'.format(filename.split('.')[0])))
                mask_image_vis = Image.fromarray(mask_miss_vis, mode='P')
                mask_image_vis.save(os.path.join(self.mask_dir, '{}_vis.png'.format(filename.split('.')[0])))
                shutil.copyfile(os.path.join(self.args.img_dir, filename),
                                os.path.join(self.image_dir, filename))

            if i % 1000 == 0:
                print("Processed {} of {}".format(i, len(self.img_ids)))

    def __coco_to_ours(self, persons):
        our_persons = list()
        for person in persons:
            dic = dict()
            dic['bbox'] = person['bbox']
            dic['kpts'] = np.zeros((18,3)).tolist()
            for i in range(17):
                dic['kpts'][COCO_TO_OURS[i]][0] = person['kpts'][i][0]
                dic['kpts'][COCO_TO_OURS[i]][1] = person['kpts'][i][1]
                dic['kpts'][COCO_TO_OURS[i]][2] = person['kpts'][i][2]
            dic['kpts'][1][0] = (person['kpts'][5][0] + person['kpts'][6][0]) * 0.5
            dic['kpts'][1][1] = (person['kpts'][5][1] + person['kpts'][6][1]) * 0.5
            if person['kpts'][5][2] == person['kpts'][6][2]:
                dic['kpts'][1][2] = person['kpts'][5][2]
            elif person['kpts'][5][2] == -1 or person['kpts'][6][2] == -1:
                dic['kpts'][1][2] = -1
            else:
                dic['kpts'][1][2] = 0

            our_persons.append(dic)

        return our_persons


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default=None, type=str,
                        dest='root_dir', help='The directory to save the ground truth.')
    parser.add_argument('--ori_anno_file', default=None, type=str,
                        dest='anno_file', help='The annotations file of coco keypoints.')
    parser.add_argument('--ori_img_dir', default=None, type=str,
                        dest='img_dir', help='The image dir corresponding to coco anno file.')

    args = parser.parse_args()

    coco_pose_generator = CocoPoseGenerator(args)
    coco_pose_generator.generate_label()
