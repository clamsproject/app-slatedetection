## run in a docker with data directory mounted with mmif, and video subdirectorys
## -data
## --mmif
## --- cpb-aacip-29-01pg4g2x.h264.mmif
## --video
## --- cpb-aacip-29-01pg4g2x.h264

import sys
import json
from app import SlateDetection
from datetime import datetime

st = datetime.now()
sd = SlateDetection()
with open("/data/mmif/cpb-aacip-29-01pg4g2x.h264.mp4.mmif", 'r') as in_f:
    mmif_json = json.load(in_f)
c = sd.annotate(mmif_json)
print (c)
