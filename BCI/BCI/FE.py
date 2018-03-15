import numpy as np
from sklearn.decomposition import PCA
import pickle
import time, os

import BCI
# feature extraction

def make_n_feature_using_pca(file, n_feature=64):
  print(file + ' file processing...')
  pca = PCA(n_components = n_feature)
  file = open(file)
  lines = file.readlines()
  file.close()
  x = []
  for line in lines:
    sl = line.split(',')
    cur_x = []
    for val in sl:
      cur_x.append(float(val))
    x.append(cur_x)
  x = np.array(x)
  x = np.transpose(x)
  st = time.time()
  reduced_x = pca.fit_transform(x)
  et = time.time()
  print((et - st))  
  return reduced_x

def n_feature_batch(path, n_feature=64):
  files = os.listdir(path)
  ff = {}
  for file in files:
    if file.split('_')[-1] == 'x.csv':
      rt = make_n_feature_using_pca(path + '/' + file, n_feature)
      if file.split('_')[0] not in ff:
        ff[file.split('_')[0]] = {}
      ff[file.split('_')[0]]['x'] = rt
    elif file.split('_')[-1] == 'y.csv':
      fy = open(path+'/'+file)
      lines = fy.readlines()
      fy.close()
      y = []
      for line in lines:
        sl = line.split(',')
        cur_y = []
        for val in sl:
          cur_y.append(float(val))
        y.append(cur_y)
      y = np.array(y); y = np.transpose(y)
      if file.split('_')[0] not in ff:
        ff[file.split('_')[0]] = {}
      ff[file.split('_')[0]]['y'] = y
  fff = open('res/'+str(n_feature)+'.pic', 'wb')
  pickle.dump(ff, fff)
  fff.close()

if __name__=='__main__':
  #n_feature_batch('dat/', n_feature=2048)
  BCI.routine_from_npy(64)