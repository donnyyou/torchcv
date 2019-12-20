#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Evaluation of coco.


import os
import sys
import argparse
import json
import numpy as np
import pickle

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from lib.tools.util.configer import Configer
from lib.tools.util.logger import Logger as Log


class VOCEvaluator(object):

    def __init__(self, configer):
        self.configer = configer

    def relabel(self, json_dir):
        submission_dir = os.path.join(json_dir, self.configer.get('method'))
        if not os.path.exists(submission_dir):
            assert os.path.exists(json_dir)
            os.makedirs(submission_dir)

        img_shotname_list = list()
        object_list = list()

        for json_file in os.listdir(json_dir):
            if 'json' not in json_file:
                continue

            json_path = os.path.join(json_dir, json_file)
            shotname, extensions = os.path.splitext(json_file)
            img_shotname_list.append(shotname)

            with open(json_path, 'r') as json_stream:
                info_tree = json.load(json_stream)
                for object in info_tree['objects']:
                    # 0-indexing
                    object_list.append([shotname, object['label'], object['score'],
                                        int(object['bbox'][0]) + 1, int(object['bbox'][1]) + 1,
                                        int(object['bbox'][2]) + 1, int(object['bbox'][3]) + 1])

        file_header_list = list()
        for i in range(len(self.configer.get('details', 'name_seq'))):
            cls = self.configer.get('details', 'name_seq')[i]
            Log.info('Writing {:s} VOC results file'.format(cls))
            filename = self.get_voc_results_file_template(submission_dir, cls)
            file_header = open(filename, 'wt')
            file_header_list.append(file_header)

        for object in object_list:
            file_header_list[object[1]].write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                object[0], object[2], object[3], object[4], object[5], object[6]))

        for file_header in file_header_list:
            file_header.close()

        Log.info('Evaluate {} images...'.format(len(img_shotname_list)))
        return submission_dir

    @staticmethod
    def get_voc_results_file_template(result_dir, cls):
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'comp3_det_test_{}.txt'.format(cls)
        path = os.path.join(result_dir, filename)
        return path

    @staticmethod
    def parse_rec(filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            objects.append(obj_struct)

        return objects

    def evaluate(self, pred_dir=None, gt_dir=None, use_07=True):
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        for i, cls in enumerate(self.configer.get('details', 'name_seq')):
            filename = self.get_voc_results_file_template(pred_dir, cls)
            rec, prec, ap = self.voc_eval(
                filename, gt_dir, cls,
                ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))

        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('--------------------------------------------------------------')

    @staticmethod
    def voc_ap(rec, prec, use_07_metric=True):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    @staticmethod
    def voc_eval(det_file,
                 gt_dir,
                 classname,
                 ovthresh=0.5,
                 use_07_metric=True):
        """rec, prec, ap = voc_eval(detpath,
                               annopath,
                               imagesetfile,
                               classname,
                               [ovthresh],
                               [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
       detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
       annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
       (default True)
    """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file
        # first load gt
        cachedir = '/tmp/voc_cache'
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)

        cachefile = os.path.join(cachedir, 'annots.pkl')
        # read list of images
        with open(os.path.join(gt_dir, 'ImageSets/Main', 'test.txt'), 'r') as f:
            lines = f.readlines()

        imagenames = [x.strip() for x in lines]
        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                annopath = os.path.join(gt_dir, 'Annotations', '%s.xml')
                recs[imagename] = VOCEvaluator.parse_rec(annopath % (imagename))
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))

            # save
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        with open(det_file, 'r') as f:
            lines = f.readlines()

        if any(lines) == 1:
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                           (BBGT[:, 2] - BBGT[:, 0]) *
                           (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = VOCEvaluator.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.

        return rec, prec, ap


if __name__ == "__main__":
    # Example:
    # python coco_evaluator.py --config_file ../../../../configs/pose/coco/openpose_vgg19_coco_pose.conf
    #                          --json_dir ../../../results/pose/coco/test_dir/coco/json/
    #                          --gt_dir /home/donny/DataSet/MSCOCO/annotations/person_keypoints_val2017.json
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='../../../configs/det/voc/ssd300_vgg16_voc_det.conf', type=str,
                        dest='config_file', help='The configs file of pose.')
    parser.add_argument('--gt_dir', default='/home/donny/DataSet/VOC/VOCdevkit/VOC2007', type=str,
                        dest='gt_dir', help='The groundtruth annotations file of voc dets.')
    parser.add_argument('--json_dir', default='../../../val/results/det/voc/test_dir/image/json', type=str,
                        dest='json_dir', help='The json dir of predict annotations.')
    parser.add_argument('--dataset', default='VOC2007', type=str,
                        dest='dataset', help='The target dataset.')
    args = parser.parse_args()

    coco_evaluator = VOCEvaluator(Configer(config_file=args.config_file))
    if args.gt_dir is not None:
        pred_dir = coco_evaluator.relabel(args.json_dir)
        coco_evaluator.evaluate(pred_dir, args.gt_dir, use_07=(args.dataset == 'VOC2007'))

    else:
        submission_dir = coco_evaluator.relabel(args.json_dir)
        Log.info('Submisson file path: {}'.format(submission_dir))
