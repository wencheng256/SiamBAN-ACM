from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamban.models.head.ban import DepthwiseBAN, MultiBAN


BANS = {
        'DepthwiseBAN': DepthwiseBAN,
        'MultiBAN': MultiBAN
       }


def get_ban_head(name, **kwargs):
    return BANS[name](**kwargs)

