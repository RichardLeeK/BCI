# Common Spatial Pattern implementation in Python, used to build spatial filters for identifying task-related activity.
import numpy as np
import scipy.linalg as la
import time
from itertools import chain
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import MIBIF

#region No Change!
# CSP takes any number of arguments, but each argument must be a collection of trials associated with a task
# That is, for N tasks, N arrays are passed to CSP each with dimensionality (# of trials of task N) x (feature vector)
# Trials may be of any dimension, provided that each trial for each task has the same dimensionality,
# otherwise there can be no spatial filtering since the trials cannot be compared


def CSP(*tasks):
	tasks = np.reshape(tasks[0], (len(tasks[0]), 5, 5, len(tasks[0][0][0])))
	if len(tasks) < 2:
		print ("Must have at least 2 tasks for filtering.")
		return (None,) * len(tasks)
	else:
		filters = ()
		# CSP algorithm
		# For each task x, find the mean variances Rx and not_Rx, which will be used to compute spatial filter SFx
		iterator = range(0,len(tasks))
		for x in iterator:
			# Find Rx
			Rx = covarianceMatrix(tasks[x][0])
			for t in range(1,len(tasks[x])):
				Rx += covarianceMatrix(tasks[x][t])
			Rx = Rx / len(tasks[x])

			# Find not_Rx
			count = 0
			not_Rx = Rx * 0
			for not_x in [element for element in iterator if element != x]:
				for t in range(0,len(tasks[not_x])):
					not_Rx += covarianceMatrix(tasks[not_x][t])
					count += 1
			not_Rx = not_Rx / count

			# Find the spatial filter SFx
			SFx = spatialFilter(Rx,not_Rx)
			filters += (SFx,)

			# Special case: only two tasks, no need to compute any more mean variances
			if len(tasks) == 2:
				filters += (spatialFilter(not_Rx,Rx),)
				break
		res = np.reshape(filters, (len(filters), 25))
		return select_max4_min4(res)


# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

# spatialFilter returns the spatial filter SFa for mean covariance matrices Ra and Rb
def spatialFilter(Ra,Rb):
	R = Ra + Rb
	E,U = la.eig(R)

	# CSP requires the eigenvalues E and eigenvector U be sorted in descending order
	ord = np.argsort(E)
	ord = ord[::-1] # argsort gives ascending order, flip to get descending
	E = E[ord]
	U = U[:,ord]

	# Find the whitening transformation matrix
	P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U))

	# The mean covariance matrices may now be transformed
	Sa = np.dot(P,np.dot(Ra,np.transpose(P)))
	Sb = np.dot(P,np.dot(Rb,np.transpose(P)))

	# Find and sort the generalized eigenvalues and eigenvector
	E1,U1 = la.eig(Sa,Sb)
	ord1 = np.argsort(E1)
	ord1 = ord1[::-1]
	E1 = E1[ord1]
	U1 = U1[:,ord1]

	# The projection matrix (the spatial filter) may now be obtained
	SFa = np.dot(np.transpose(U1),P)
	
	return SFa.astype(np.float32)

def select_max3_min3(csp):
  res = []
  for x in csp:
    min = [10000, 10000, 10000]
    max = [0, 0, 0]
    for v in x:
      if v > max[0]:
        max[2] = max[1]
        max[1] = max[0]
        max[0] = v
      elif v > max[1]:
        max[2] = max[1]
        max[1] = v
      elif v > max[2]:
        max[2] = v

      if v < min[0]:
        min[2] = min[1]
        min[1] = min[0]
        min[0] = v
      elif v < min[1]:
        min[2] = min[1]
        min[1] = v
      elif v < min[2]:
        min[2] = v
    max.extend(min)
    res.append(max)
  return np.array(res)

def select_max4_min4(csp):
  res = []
  for x in csp:
    min = [10000, 10000, 10000, 10000]
    max = [0, 0, 0, 0]
    for v in x:
      if v > max[0]:
        max[3] = max[2]
        max[2] = max[1]
        max[1] = max[0]
        max[0] = v
      elif v > max[1]:
        max[3] = max[2]
        max[2] = max[1]
        max[1] = v
      elif v > max[2]:
        max[3] = max[2]
        max[2] = v
      elif v > max[3]:
        max[3] = v

      if v < min[0]:
        min[3] = min[2]
        min[2] = min[1]
        min[1] = min[0]
        min[0] = v
      elif v < min[1]:
        min[3] = min[2]
        min[2] = min[1]
        min[1] = v
      elif v < min[2]:
        min[3] = min[2]
        min[2] = v
      elif v < min[3]:
        min[3] = v
    max.extend(min)
    res.append(max)
  return np.array(res)


def bandpass_filter(data, lowcut, highcut, fs, order=5):
  from scipy.signal import butter, lfilter
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  y = lfilter(b, a, data)
  return y
#endregion

def arr_bandpass_filter(data, lowcut, highcut, fs, order=5):
  y = np.array(data)
  filtered_y = np.zeros(data.shape)
  for i in range(len(data)):
    for j in range(len(data[i])):
      cur_data = data[i][j]
      cur_y = bandpass_filter(cur_data, lowcut, highcut, fs, order)
      y[i][j] = cur_y
  return y

def filterbank_sub_method(x, i, fs, res):
  y = bandpass_filter(x, i * 4, (i + 1) * 4, fs)
  y = CSP(y)
  res[i - 1] = y

def filterbank_CSP(x, fs=250):
  res = [[] for j in range(9)]
  ps = []
  pool = ThreadPoolExecutor(max_workers=3)
  for i in range(1, 10):
    p = pool.submit(filterbank_sub_method, x, i, fs, res)
    ps.append(p)
  for p in as_completed(ps):
    p = None
  y_s = np.array(res)
  y_s = np.reshape(y_s, (len(y_s[0]), len(y_s), len(y_s[0][0])))
  return y_s

def filterbank_CSP_original(x, fs=250):
  res = [[] for j in range(9)]
  ps = []
  for i in range(1, 10):
    filterbank_sub_method(x, i, fs, res)
  y_s = np.array(res)
  y_s = np.reshape(y_s, (len(y_s[0]), len(y_s), len(y_s[0][0])))
  return y_s

def multi_test():
  import BCI
  x, y = load_data()
  kv = BCI.gen_kv_idx(y, 10)
  for train_idx, test_idx in kv:
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    import time
    start_time = time.time() 
    fb_csp = filterbank_CSP_original(x_train)
    print("--- %s seconds (single)---" %(time.time() - start_time))

    start_time = time.time() 
    fb_csp = filterbank_CSP(x_train)
    print("--- %s seconds (multi)---" %(time.time() - start_time))

def spectral_temporal_CSP(x, fs=250):
  y_s = []
  for i in range(1, 10):
    y = bandpass_filter(x, i * 4, (i + 1) * 4, fs)
    step = int(len(y)/5)
    for j in range(5):
      y_ = CSP(y[:,:,j * step: (j + 1) * step])
      y_s.append(y_)
  y_s = np.array(y_s)
  y_s = np.reshape(y_s, (len(y_s[0]), len(y_s), len(y_s[0][0])))
  return y_s

def temporal_spectral_CSP_sub_method(y, j, fs, res):
  y_ = bandpass_filter(y, j * 4, (j + 1)* 4, fs)
  y_ = CSP(y_)
  res[j] = y_

def temporal_spectral_CSP(x, fs=250):
  res = [[] for i in range(45)]
  step = int(len(x)/5)
  ps = []
  pool = ThreadPoolExecutor(max_workers=9)
  for i in range(5):
    y = x[:, :, i * step : (i+1) * step]
    for j in range(1, 10):
      p = pool.submit(temporal_spectral_CSP_sub_method, y, j, fs, res)
      ps.append(p)
  for p in as_completed(ps):
    p = None
  y_s = np.array(res)
  y_s = np.reshape(y_s, (len(y_s[0]), len(y_s), len(y_s[0][0])))
  return y_s

def temporal_spectral_CSP_original(x, fs=250):
  y_s = []
  step = int(len(x)/5)
  for i in range(5):
    y = x[:, :, i * step : (i+1) * step]
    for j in range(1, 10):
      y_ = bandpass_filter(y, j * 4, (j + 1)* 4, fs)
      y_ = CSP(y_)
      y_s.append(y_)
  y_s = np.array(y_s)
  y_s = np.reshape(y_s, (len(y_s[0]), len(y_s), len(y_s[0][0])))
  return y_s

def load_data(path='data/A01T.npz'):
  import Competition, BCI
  x, y = Competition.load_one_data(path)
  x = np.array(x)
  y = np.array(BCI.lab_inv_translator(y, 4))
  return x, y

def data_trans_tmp(res):
  new_res = {};  new_res['csp'] = {};  new_res['fbcsp'] = {};
  new_res['tscsp'] = {};  new_res['stcsp'] = {}
  new_res['csp']['train_x'] = res['csp']['train_x']
  new_res['csp']['test_x'] = res['csp']['test_x']
  new_res['fbcsp']['train_x'] = np.reshape(res['fbcsp']['train_x'], (249, 54))
  new_res['fbcsp']['test_x'] = np.reshape(res['fbcsp']['test_x'], (24, 54))
  new_res['stcsp']['train_x'] = np.reshape(res['stcsp']['train_x'], (249, 45*6))
  new_res['stcsp']['test_x'] = np.reshape(res['stcsp']['test_x'], (24, 45*6))
  new_res['tscsp']['train_x'] = np.reshape(res['tscsp']['train_x'], (249, 45*6))
  new_res['tscsp']['test_x'] = np.reshape(res['tscsp']['test_x'], (24, 45*6))
  new_res['train_y'] = res['y_train']; new_res['test_y'] = res['y_test']
  return new_res

def one_set(clf, res):
  clf[0].fit(res['csp']['train_x'], res['y_train'])
  clf[1].fit(np.reshape(res['fbcsp']['train_x'], (249, 54)), res['y_train'])
  clf[2].fit(np.reshape(res['stcsp']['train_x'], (249, 45*6)), res['y_train'])
  clf[3].fit(np.reshape(res['tscsp']['train_x'], (249, 45*6)), res['y_train'])
  print(clf[0].score(res['csp']['test_x'], res['y_test']))
  print(clf[1].score(np.reshape(res['fbcsp']['test_x'], (24, 54)), res['y_test']))
  print(clf[2].score(np.reshape(res['stcsp']['test_x'], (24, 45*6)), res['y_test']))
  print(clf[3].score(np.reshape(res['tscsp']['test_x'], (24, 45*6)), res['y_test']))

def mibif(x, y, order = None):
  xs = [[], [], [], []]
  mi = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  for i in range(len(y)):
    xs[y[i]].append(x[i])
  vs_arr = np.array(4, 10, 6)

def total_csp_save():
  import os, BCI
  files = os.listdir('data')
  for file in files:
    if file[-5] != 'T': continue
    print(file + ' processing...')
    x, y = load_data('data/' + file)
    kv = BCI.gen_kv_idx(y, 10)
    res = {}
    i = 0
    for train_idx, test_idx in kv:
      res[i] = {}
      x_train, y_train = x[train_idx], y[train_idx]
      x_test, y_test = x[test_idx], y[test_idx]
      print(str(i) + ' csp ...')
      csp_train = CSP(x_train)
      csp_test = CSP(x_test)
      print(str(i) + ' fbcsp ...')
      fb_csp_train = filterbank_CSP_original(x_train)
      fb_csp_test = filterbank_CSP_original(x_test)
      print(str(i) + ' stcsp ...')
      st_csp_train = spectral_temporal_CSP(x_train)
      st_csp_test = spectral_temporal_CSP(x_test)
      print(str(i) + ' tscsp ...')
      ts_csp_train = temporal_spectral_CSP_original(x_train)
      ts_csp_test = temporal_spectral_CSP_original(x_test)
      print(str(i) + ' saving ...')
      res[i]['train'] = {}
      res[i]['train']['csp'] = csp_train
      res[i]['train']['fbcsp'] = fb_csp_train
      res[i]['train']['stcsp'] = st_csp_train
      res[i]['train']['tscsp'] = ts_csp_train
      res[i]['train']['x'] = x_train
      res[i]['train']['y'] = y_train
      res[i]['test'] = {}
      res[i]['test']['csp'] = csp_test
      res[i]['test']['fbcsp'] = fb_csp_test
      res[i]['test']['stcsp'] = st_csp_test
      res[i]['test']['tscsp'] = ts_csp_test
      res[i]['test']['x'] = x_test
      res[i]['test']['y'] = y_test
      i += 1
    import pickle
    with open('csp_10_data/' + file + '.pic', 'wb') as f:
      pickle.dump(res, f)


  
def load_csp(count):
  
  with open('csp_data/A0' + str(count) + 'T.npz.pic', 'rb') as f:
    res = pickle.load(f)
  for i in range(5):
    print(i)
    x_train = res[i]['train']['x']
    y_train = res[i]['train']['y']
    y_test = res[i]['test']['y']
    fb_csp = res[i]['train']['fbcsp']
    fb_mi = MIBIF.fb_mibif_with_csp(x_train, y_train, fb_csp)
    fb_idx = np.argmin(fb_mi)
    fb_x = MIBIF.mi_selector(fb_csp, fb_idx)
    fb_x_t = MIBIF.mi_selector(res[i]['test']['fbcsp'], fb_idx)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    fb_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    fb_lda.fit(fb_x, y_train.argmax(axis=1))
    fb_score = fb_lda.score(fb_x_t, y_test.argmax(axis=1))

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(res[i]['train']['csp'], y_train.argmax(axis=1))
    score = fb_lda.score(res[i]['test']['csp'], y_test.argmax(axis=1))

    ts_csp = res[i]['train']['tscsp']
    st_csp = res[i]['train']['stcsp']

    ts_mi = MIBIF.fb_mibif_with_csp(x_train, y_train, ts_csp)
    st_mi = MIBIF.fb_mibif_with_csp(x_train, y_train, st_csp)
    
    ts_idx = np.argmin(ts_mi)
    st_idx = np.argmin(st_mi)
    
    ts_x = MIBIF.mi_selector(ts_csp, ts_idx)
    st_x = MIBIF.mi_selector(st_csp, st_idx)

    ts_x_t = MIBIF.mi_selector(res[i]['test']['tscsp'], ts_idx)
    st_x_t = MIBIF.mi_selector(res[i]['test']['stcsp'], st_idx)

    ts_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    ts_lda.fit(ts_x, y_train.argmax(axis=1))
    ts_score = ts_lda.score(ts_x_t, y_test.argmax(axis=1))

    st_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    st_lda.fit(fb_x, y_train.argmax(axis=1))
    st_score = fb_lda.score(fb_x_t, y_test.argmax(axis=1))

    pen = open('csp_res_' + str(count) + '.csv', 'a')
    pen.write(str(i) + ',' + str(score) + ',' + str(fb_score) + ',' + str(ts_score) + ',' + str(st_score) + '\n')


def test2():
  import pickle
  with open('tmp_data/tmp.pic', 'rb') as f:
    res = pickle.load(f)
  res = data_trans_tmp(res)
  import FFS
  feat_selector = FFS.FilterFeatureSelection(res['tscsp']['train_x'], res['train_y'])
  feat_selector.change_method('MIFS')
  selected_features = feat_selector.run(50)
  
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
  clf.fit(res['tscsp']['train_x'][:,selected_features], res['train_y'])
  score = clf.score(res['tscsp']['test_x'][:,selected_features], res['test_y'])
  print(score)

  print('abc')


def test():
  import Competition, BCI
  x, y = Competition.load_one_data('data/A01T.npz')
  x = np.array(x)
  y = np.array(BCI.lab_inv_translator(y, 4))
  kv = BCI.gen_kv_idx(y, 10)
  accuracies = []
  accuracies_fb = []
  accuracies_st = []
  accuracies_ts = []
  for train_idx, test_idx in kv:
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    start_time = time.time()
    csp_train = CSP(x_train)
    csp_test = CSP(x_test)
    print("CSP seconds: %", (time.time() - start_time))

    start_time = time.time()
    fb_csp_train = filterbank_CSP(x_train)
    fb_csp_test = filterbank_CSP(x_test)
    print("FBCSP seconds: %", (time.time() - start_time))

    start_time = time.time()
    st_csp_train = spectral_temporal_CSP(x_train)
    st_csp_test = spectral_temporal_CSP(x_test)
    print("STCSP seconds: %", (time.time() - start_time))

    start_time = time.time()
    ts_csp_train = temporal_spectral_CSP(x_train)
    ts_csp_test = temporal_spectral_CSP(x_test)
    print("TSCSP seconds: %", (time.time() - start_time))

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf_csp = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_fb = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_st = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_ts = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    res = {}
    res['csp'] = {}
    res['fbcsp'] = {}
    res['stcsp'] = {}
    res['tscsp'] = {}
    res['csp']['train_x'] = csp_train
    res['csp']['test_x'] = csp_test
    res['fbcsp']['train_x'] = fb_csp_train
    res['fbcsp']['test_x'] = fb_csp_test
    res['stcsp']['train_x'] = st_csp_train
    res['stcsp']['test_x'] = st_csp_test
    res['tscsp']['train_x'] = ts_csp_train
    res['tscsp']['test_x'] = ts_csp_test
    res['y_train'] = y_train.argmax(axis=1)
    res['y_test'] = y_test.argmax(axis=1)
    import pickle
    with open('tmp_data/tmp.pic', 'wb') as f:
      pickle.dump(res, f)

    cur_csp_x_test = pca_csp.transform(cur_csp_x_test)
    cur_fbcsp_x_test = pca_fb.transform(cur_fbcsp_x_test)
    cur_stcsp_x_test = pca_st.transform(cur_stcsp_x_test)
    cur_tscsp_x_test = pca_ts.transform(cur_tscsp_x_test)

    clf_csp.fit(cur_csp_x_train, y_train.argmax(axis=1))
    clf_fb.fit(cur_fbcsp_x_train, y_train.argmax(axis=1))
    clf_st.fit(cur_stcsp_x_train, y_train.argmax(axis=1))
    clf_ts.fit(cur_tscsp_x_train, y_train.argmax(axis=1))

    csp_score = clf_csp.score(cur_csp_x_test, y_test.argmax(axis=1))
    fbcsp_score = clf_fb.score(cur_fbcsp_x_test, y_test.argmax(axis=1))
    stcsp_score = clf_st.score(cur_stcsp_x_test, y_test.argmax(axis=1))
    tscsp_score = clf_ts.score(cur_tscsp_x_test, y_test.argmax(axis=1))

    print(csp_score)
    print(fbcsp_score)
    print(stcsp_score)
    print(tscsp_score)

    accuracies.append(csp_score)
    accuracies_fb.append(fbcsp_score)
    accuracies_st.append(stcsp_score)
    accuracies_ts.append(tscsp_score)

  pen = open('csp_res.csv', 'w')
  sentence = 'CSP'
  for v in accuracies:
    sentence += ',' + str(v)
  pen.write(sentence + '\n')
  sentence = 'FBCSP'
  for v in accuracies_fb:
    sentence += ',' + str(v)
  pen.write(sentence + '\n')
  sentence = 'STCSP'
  for v in accuracies_st:
    sentence += ',' + str(v)
  pen.write(sentence + '\n')
  sentence = 'TSCSP'
  for v in accuracies_ts:
    sentence += ',' + str(v)
  pen.write(sentence)
  pen.close()

def test_csp():
  import Competition, BCI
  x, y = Competition.load_one_data('data/A01T.npz')
  x = np.array(x)
  y = np.array(BCI.lab_inv_translator(y, 4))
  kv = BCI.gen_kv_idx(y, 10)
  for train_idx, test_idx in kv:
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    csp_train = CSP(x_train)
    csp_test = CSP(x_test)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    clf = LinearDiscriminantAnalysis()
    csp_train = np.reshape(csp_train, (len(csp_train), 25))
    clf.fit(csp_train, y_train.argmax(axis=1))

    csp_test = np.reshape(csp_test, (len(csp_test), 25))
    clf.predict(csp_test)

    print('adfd')

def numpy_to_mat(path='data/A01T.npz'):
  import Competition, BCI, scipy.io
  x, y = Competition.load_one_data(path)
  file = {}
  file['clab'] = [i for i in range(25)]
  file['fs'] = 250
  file['y'] = np.transpose(BCI.lab_inv_translator(y, 4))
  #file['x'] = bandpass_filter(x, 5, 30, 250)
  file['x'] = x
  file['t'] = [i for i in range(750)]
  file['className'] = np.transpose(['a', 'b', 'c', 'd'])
  
  scipy.io.savemat(path.split('/')[-1].split('.')[0] + '.mat', file)

def new_numpy_to_mat(path='data/A01T.npz'):
  import Competition, BCI, scipy.io
  x, y = Competition.load_one_data(path)
  file = [{'clab':[], 'fs':250, 'x': [], 'y': []}, {'clab':[], 'fs':250, 'x': [], 'y': []}, {'clab':[], 'fs':250, 'x': [], 'y': []}]
  file[0]['clab'] = ['01', '23']
  file[1]['clab'] = ['02', '13']
  file[2]['clab'] = ['03', '12']
  for i in range(len(y)):
    if y[i] == 0:
      file[0]['x'].append(x[i]); file[0]['y'].append([1, 0])
      file[1]['x'].append(x[i]); file[1]['y'].append([1, 0])
      file[2]['x'].append(x[i]); file[2]['y'].append([1, 0])
    elif y[i] == 1:
      file[0]['x'].append(x[i]); file[0]['y'].append([1, 0])
      file[1]['x'].append(x[i]); file[1]['y'].append([0, 1])
      file[2]['x'].append(x[i]); file[2]['y'].append([0, 1])
    elif y[i] == 2:
      file[0]['x'].append(x[i]); file[0]['y'].append([0, 1])
      file[1]['x'].append(x[i]); file[1]['y'].append([1, 0])
      file[2]['x'].append(x[i]); file[2]['y'].append([0, 1])
    elif y[i] == 3:
      file[0]['x'].append(x[i]); file[0]['y'].append([0, 1])
      file[1]['x'].append(x[i]); file[1]['y'].append([0, 1])
      file[2]['x'].append(x[i]); file[2]['y'].append([1, 0])
  scipy.io.savemat('mat/' + path.split('/')[-1].split('.')[0] + '_1.mat', file[0])
  scipy.io.savemat('mat/' + path.split('/')[-1].split('.')[0] + '_2.mat', file[1])
  scipy.io.savemat('mat/' + path.split('/')[-1].split('.')[0] + '_3.mat', file[2])


def load_new_mat():
  import BCI, scipy.io, Competition
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  from sklearn.naive_bayes import GaussianNB
  from sklearn.metrics import cohen_kappa_score
  aa = scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/res/A01T_CSP.mat')
  x, y = Competition.load_one_data('data/A01T.npz')
  y = np.array(BCI.lab_inv_translator(y, 4))
  x_ = np.transpose(aa['csp_res'])
  kv = BCI.gen_kv_idx(y, 10)
  for train_idx, test_idx in kv:
    x_train, y_train = x_[train_idx], y[train_idx]
    x_test, y_test = x_[test_idx], y[test_idx]
    clf = LinearDiscriminantAnalysis()
    x_train = clf.fit_transform(x_train, y_train.argmax(axis=1))
    x_test = clf.transform(x_test)
    clf2 = GaussianNB()
    clf2.fit(x_train, y_train.argmax(axis=1))
    y_predict = clf2.predict(x_test)
    
    cohen_score = cohen_kappa_score(y_predict, y_test.argmax(axis=1))
    score = clf2.score(x_test, y_test.argmax(axis=1))
    pen = open("cohen_score7.csv", 'a')
    pen.write(str(cohen_score) + ',' + str(score) + '\n')
  print('abc')

def test_csp_old():
  import json
  with open('test2.js', 'r') as f:
    data = json.load(f)
  import BCI
  for k in data:
    x = np.array(data[k]['x'])
    y = data[k]['y']
    y = np.array(BCI.lab_inv_translator(y))
    kv = BCI.gen_kv_idx(y, 10)
  for train_idx, test_idx in kv:
    x_train, y_train = x[train_idx], y[train_idx]
    new_x = [[], [], []]
    y_idx = np.where(y_train)[1]
    for i in range(len(y_idx)):
      new_x[int(y_idx[i]) - 1].append(x_train[i])
    vvv = CSP(new_x[0], new_x[1], new_x[2])
    x_test, y_test = x[test_idx], y[test_idx]
    
    print(a)
    return a

if __name__ == '__main__':
  load_new_mat()
  #new_numpy_to_mat()
  #load_mat()
  #test_num_2()
  #a = test_csp()
  """
  import threading
  ts = []
  for i in range(1, 10):
    t = threading.Thread(target=load_csp, args=([i]))
    t.start()
    ts.append(t)
  for t in ts:
    t.join()
  """
  #total_csp_save()