# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys

import cv2
import torch
import vot_tool as vot

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.bbox import corner2center, Corner
from siamban.utils.model_load import load_pretrain

parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--config', default='config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='model.pth', type=str, help='config file')

args = parser.parse_args()

torch.set_num_threads(1)

def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    handle = vot.VOT("rectangle")
    region = handle.region()

    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    img = cv2.imread(imagefile)
    left = max(region.x, 0)
    top = max(region.y, 0)

    right = min(region.x + region.width, img.shape[1] - 1)
    bottom = min(region.y + region.height, img.shape[0] - 1)

    cx, cy, w, h = corner2center(Corner(left, top, right, bottom))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    tracker.init(img, gt_bbox_)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.imread(imagefile)
        outputs = tracker.track(image)
        pred_bbox = outputs['bbox']
        conf = outputs['best_score']


        handle.report(vot.Rectangle(*pred_bbox), conf)

if __name__ == '__main__':
    main()
