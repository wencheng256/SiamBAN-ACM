from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.xcorr import xcorr_fast, xcorr_depthwise

class BAN(nn.Module):
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f, bbox):
        raise NotImplementedError

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=3, bias=True, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)
                )

        self.xorr_bbox = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, hidden, bias=False)
        )

        self.xorr_search = nn.Conv2d(hidden, hidden, kernel_size=5, bias=False)
        self.xorr_kernel = nn.Conv2d(hidden, hidden, kernel_size=5, bias=True)

        self.xorr_activate = nn.Sequential(
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )

    def init(self, kernel, bbox):
        kernel = self.conv_kernel(kernel)
        kernel_part = self.xorr_kernel(kernel)
        bbox_part = self.xorr_bbox(bbox[:, 2:]).view(*kernel_part.shape)
        self.kernel_part = kernel_part
        self.bbox_part = bbox_part

    def track(self, search):
        search = self.conv_search(search)
        search_part = self.xorr_search(search)
        feature = self.xorr_activate(search_part + self.kernel_part + self.bbox_part)
        out = self.head(feature)
        return out


class DepthwiseBAN(BAN):
    def __init__(self, in_channels=256, out_channels=256, cls_out_channels=2, weighted=False):
        super(DepthwiseBAN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, cls_out_channels)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4)

    def init(self, z_f, bbox):
        self.cls.init(z_f, bbox)
        self.loc.init(z_f, bbox)

    def track(self, x_f):
        cls = self.cls.track(x_f)
        loc = self.loc.track(x_f)
        return cls, loc


class MultiBAN(BAN):
    def __init__(self, in_channels, cls_out_channels, weighted=False):
        super(MultiBAN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('box'+str(i+2), DepthwiseBAN(in_channels[i], in_channels[i], cls_out_channels))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
        self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))

    def init(self, z_fs, bbox):
        for idx, z_f in enumerate(z_fs, start=2):
            box = getattr(self, 'box'+str(idx))
            box.init(z_f, bbox)

    def track(self, x_fs):
        cls = []
        loc = []
        for idx, x_f in enumerate(x_fs, start=2):
            box = getattr(self, 'box'+str(idx))
            c, l = box.track(x_f)
            cls.append(c)
            loc.append(torch.exp(l*self.loc_scale[idx-2]))

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)