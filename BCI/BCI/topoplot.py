import os
from PIL import Image
import numpy as np
import itertools
import pickle
import RCNN
import BCI
"""
path = 'C:/Users/Richard/Documents/MATLAB/fig2/'

files = os.listdir(path)
dic = {}
for file in files:
  print(file)
  im_frame = Image.open(path+file)
  np_frame = np.array(im_frame.getdata())
  sf = file.split('_')
  if sf[0] not in dic:
    dic[sf[0]] = {'x': [], 'y': []}
  dic[sf[0]]['x'].append(np_frame)
  dic[sf[0]]['y'].append(int(sf[3]))

for k, v in dic.items():
  f = open('fig_dat2/' + k + '_topo-polt.pic', 'wb')
  pickle.dump(v, f)
  f.close()
"""
f = open('fig_dat/bwyu_topo-polt.pic', 'rb')
dat = pickle.load(f)
print('abc')

model = RCNN.create_model()
BCI.kf