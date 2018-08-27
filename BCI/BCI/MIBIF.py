# Mutual information based individual feature (MIBIF)
import CSP, BCI, utils
import numpy as np
from multiprocessing import Process

def m_mutual_information(f1, f2, i, j, k, mis):
  print(str(i) + '_' + str(j) + '_' + str(k) + ' processing...')
  mis[i][k][j] = mutual_information(f1, f2)

def mutual_information(f1, f2):
  merge = np.concatenate((f1, f2), axis=0)
  estimated_density = parzen_kde(merge, merge, 1)
  entropy = -np.sum(np.log(estimated_density)) / len(estimated_density)

  class_one_density = parzen_kde(f1, f1, 1)
  class_two_density = parzen_kde(f2, f2, 1)

  hac_one = -np.sum(np.log(class_one_density)) / len(class_one_density)
  hac_two = -np.sum(np.log(class_two_density)) / len(class_two_density)

  cond_entropy = (hac_one + hac_two) / 2
  return entropy - cond_entropy

def parzen_kde(train, test, window):
  from scipy.stats import multivariate_normal
  train_size = np.shape(train)[0]
  test_size = np.shape(test)[0]
  #num_feature = len(train[1])
  num_feature = 1
  covariance = np.zeros((num_feature, num_feature))
  for i in range(num_feature):
    covariance[i][i] = np.var(train)
  estimated_density = np.zeros((test_size, 1))
  for i in range(len(test)):
    x = test[i]
    test_sample_matrix = np.ones((train_size, 1)) * x
    new_diff = test_sample_matrix - np.reshape(train, (len(train), 1))
    for j in range(num_feature):
      new_diff[abs(new_diff[:, j]) > window, j] = 10000000
    mvn = multivariate_normal(np.zeros((1, num_feature)), covariance)
    estimated_density[i] = np.mean((1/(window**num_feature)) * mvn.pdf((new_diff/window)))
  return estimated_density


def fb_mibif(x, y):
  fb_csp = CSP.filterbank_CSP(x)
  xs = [[], [], [], []]
  for i in range(len(fb_csp)):
    xs[y.argmax(axis=1)[i]].append(fb_csp[i])
  mis = np.zeros((len(xs), len(xs[0][0]), len(xs[0]) + 10))
  for i in range(len(xs)): # class number
    for j in range(len(xs[i])): # epoch count
      for k in range(len(xs[i][j])): # filter number
        one = xs[i][j][k]
        rest = []
        for l in rnage(len(xs)):
          if i == l: continue
          try: rest.extend(xs[l][j][k])
          except Exception as e: print(e)
        mis[i][j][k] = mutual_information(one, rest)
  return np.sum(np.sum(mis, axis=2), axis=0)

def fb_mibif_with_csp_old(x, y, fb_csp):
  xs = [[], [], [], []]
  for i in range(np.shape(fb_csp)[1]):
    xs[y[i]].append(np.transpose(fb_csp[:,i,:]))
  mis = np.zeros((len(xs), len(xs[0][0]), len(xs[0]) + 10))
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
        mis[i][k][j] = mutual_information(one, rest)
  return np.sum(np.sum(mis, axis=2), axis=0).argmax()

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




def fb_mibif_opt(x, y):
  fb_csp = CSP.filterbank_CSP(x)
  xs = [[], [], [], []]
  for i in range(len(fb_csp)):
    xs[y.argmax(axis=1)[i]].append(fb_csp[i])
  mis = np.zeros((len(xs), len(xs[0][0]), len(xs[0]) + 10))
  for i in range(len(xs)): # class number
    for j in range(len(xs[i])): # epoch count
      for k in range(len(xs[i][j])): # filter number
        one = xs[i][j][k]
        rest = []
        for l in rnage(len(xs)):
          if i == l: continue
          try: rest.extend(xs[l][j][k])
          except Exception as e: print(e)
        mis[i][j][k] = mutual_information(one, rest)
  return np.sum(np.sum(mis, axis=2), axis=0)


def ts_mibif(x, y):
  fb_csp = CSP.temporal_spectral_CSP(x)
  xs = [[], [], [], []]
  for i in range(len(fb_csp)):
    xs[y.argmax(axis=1)[i]].append(fb_csp[i])
  mis = np.zeros((len(xs), len(xs[0][0]), len(xs[0]) + 10))
  for i in range(len(xs)): # class number
    for j in range(len(xs[0][0])): # filter number
      for k in range(len(xs[i])): # epoch count
        one = xs[i][k][j]
        rest = []
        for l in range(len(xs)):
          if i == l: continue
          try:
            rest.extend(xs[l][k][j])
          except Exception as e:
            print(e)
        mis[i][j][k] = mutual_information(one, rest)
  return np.sum(np.sum(mis, axis=2), axis=0)

def mi_selector(x, idx):
  new_x = []
  for v in x:
    new_x.append(v[idx])
  return np.array(new_x)

def batch():
  x, y = CSP.load_data()
  kv = BCI.gen_kv_idx(y, 10)
  for train_idx, test_idx in kv:
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    fb_mi = fb_mibif(x_train, y_train)
    #ts_mi = ts_mibif(x_train, y_train)

    fb_idx = np.argmax(fb_mi)
    #ts_idx = np.argmax(ts_mi)

    fb_csp = CSP.filterbank_CSP(x_train)
    #ts_csp = CSP.temporal_spectral_CSP(x_train)

    fb_x = mi_selector(fb_csp, fb_idx)
    ts_x = mi_selector(ts_csp, fb_idx)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    fb_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    ts_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    fb_lda.fit(fb_x, y_train)
    ts_lad.fit(ts_x, y_train)
    
    fb_x_t = mi_selector(CSP.filterbank_CSP(x_test), fb_x)
    #ts_x_t = mi_selector(CSP.temporal_spectral_CSP(x_test), ts_mi)

    fb_score = fb_lda.score(fb_x_t, y_test)
    ts_score = ts_lda.score(ts_x_t, y_test)

    print(fb_score)
    print(ts_score)
if __name__ == '__main__':
  batch()