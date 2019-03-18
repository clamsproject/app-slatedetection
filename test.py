import sys
import json
from slatedetect import SlateDetection
from datetime import datetime

st = datetime.now()
sd = SlateDetection()
a = open(sys.argv[1])
b = a.read()
c = sd.annotate(b)
with open("../test-jsons/slate-detect.json", "w") as f:
    f.write(str(c))

# for i in c.views:
#     a = i.__dict__
#     print (a)
#     c = a.get("contains")
#     bd = a.get("annotations")
#     for d in bd:
#         print (d.__dict__)
# print (datetime.now()-st)