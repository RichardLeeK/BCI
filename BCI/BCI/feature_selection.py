import numpy as np
from sklearn.svm import SVC
import BCI

def fb_mibif_with_csp(x, y, fb_csp):
  xs = [[], [], [], []]
  for i in range(np.shape(fb_csp)[1]):
    xs[y[i]].append(np.transpose(fb_csp[:,i,:]))
  mis = np.zeros((len(xs), len(xs[0]) + 10, len(xs[0][0])))
  for i in range(len(xs)): # class number
    print('i: ' + str(i))
    for j in range(len(xs[i])): # epoch count
      for k in range(len(xs[i][j])): # filter number
        one = xs[i][j][k]
        rest = []
        for l in range(len(xs)):
          if i == l: continue
          try: rest.extend(xs[l][j][k])
          except: continue
        mis[i][j][k] = mutual_information(one, np.array(rest))
  return np.sum(np.sum(mis, axis=1), axis=0).argmax()

def lsvm_filter(x, y):
  #x = np.array(x)[:,:,18:]
  scores = []
  for i in range(len(x[0][0])):
    sub_x = np.transpose(np.array(x[:,:,i]))
    clf = SVC(kernel='linear')
    clf.fit(sub_x, y)
    scores.append(clf.score(sub_x, y))
  return np.array(scores).argmax()

def lsvm_filter_pp(x1, y1, x2, y2):
  scores = []
  for i in range(len(x1[0][0])):
    train_x = np.transpose(np.array(x1[:,:,i]))
    valid_x = np.transpose(np.array(x2[:,:,i]))
    clf = SVC(kernel='linear')
    clf.fit(train_x, y1.argmax(axis=1))
    scores.append(clf.score(valid_x, y2.argmax(axis=1)))
  return np.array(scores).argmax()

def lsvm_filter_pp2(x, y):
  scores = []
  for i in range(len(x[0][0])):
    kv = BCI.gen_kv_idx(y, 9)
    cur_scores = []
    for train_idx, test_idx in kv:
      train_x, train_y = x[:,train_idx,:], y[train_idx]
      test_x, test_y = x[:,test_idx,:], y[test_idx]
      train_x = np.transpose(np.array(train_x[:,:,i]))
      test_x = np.transpose(np.array(test_x[:,:,i]))
      clf = SVC(kernel='linear')
      clf.fit(train_x, train_y.argmax(axis=1))
      cur_scores.append(clf.score(test_x, test_y.argmax(axis=1)))
    scores.append(np.mean(cur_scores))
  return np.array(scores).argmax()


