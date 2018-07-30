
import os
import pytz
import numpy as np
from datetime import datetime, timedelta

def fromOADate(v):
    return datetime(1899, 12, 30, 0, 0, 0, tzinfo=pytz.utc) + timedelta(days=v)

def fromDatetime(date1):
    temp = datetime(1899, 12, 30, 0, 0, 0, tzinfo=pytz.utc)    # Note, not 31st Dec but 30th!
    delta = date1 - temp
    return float(delta.days) + (float(delta.seconds) / 86400)

path = 'D:/Richard/TBI 300/TBI csv_merged original/'
outpath = 'D:/Richard/TBI 300/TBI csv_merged connect/'
print('1')

files = os.listdir(path)

for file in files:
  f = open(path + file)
  print(file)
  lines = f.readlines()
  midas = 0
  pen = open(outpath + file, 'w')
  pen.write(lines[0])
  bef_time = float(lines[1].split(',')[0])
  for line in lines[1:]:
    sl = line.split(',')
    dtime = float(sl[0])
    if (dtime-bef_time) > 0.000001:
      midas += dtime - bef_time - 0.0000006
    p_dtime = str(dtime - midas)
    sentence = p_dtime
    for v in sl[1:]:
      sentence += ',' + v
    pen.write(sentence)
    bef_time = dtime
    
     
      





