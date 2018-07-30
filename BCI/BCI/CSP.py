# Common Spatial Pattern implementation in Python, used to build spatial filters for identifying task-related activity.
import numpy as np
import scipy.linalg as la

# CSP takes any number of arguments, but each argument must be a collection of trials associated with a task
# That is, for N tasks, N arrays are passed to CSP each with dimensionality (# of trials of task N) x (feature vector)
# Trials may be of any dimension, provided that each trial for each task has the same dimensionality,
# otherwise there can be no spatial filtering since the trials cannot be compared
def CSP(*tasks):
	tasks = np.reshape(tasks[0], (len(tasks[0]), 5, 5, 1875))
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
		return filters

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

def bandpass_filter(data, lowcut, highcut, fs, order=5):
  from scipy.signal import butter, lfilter
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  y = lfilter(b, a, data)
  return y

def filterbank_CSP(x, fs=250):
  y_s = []
  for i in range(1, 10):
    y = bandpass_filter(x, i * 4, (i + 1) * 4, fs)
    y = CSP(y)
    y_s.append(y)
  return y_s

def spectral_temporal_CSP(x, fs=250):
  y_s = []
  for i in range(1, 10):
    y = bandpass_filter(x, i * 4, (i + 1) * 4, fs)
    step = int(len(y)/10)
    for j in range(step):
      y_ = CSP(y[j * step: (j + 1) * step])
      y_s.append(y_)
  return y_s

def test():
  import Competition, BCI
  x, y, = Competition.load_one_data('data/A01T.npz')
  x = np.array(x)
  y = np.array(BCI.lab_inv_translator(y, 4))
  kv = BCI.gen_kv_idx(y, 10)
  for train_idx, test_idx in kv:
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    csp_train = CSP(x_train)
    csp_test = CSP(x_test)

    fb_csp_train = filterbank_CSP(x_train)
    fb_csp_test = filterbank_CSP(x_test)

    st_csp_train = spectral_temporal_CSP(x_train)
    st_csp_train = spectral_temporal_CSP(x_test)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf_csp = LinearDiscriminantAnalysis(shrinkage='auto')
    clf_fb = LinearDiscriminantAnalysis(shrinkage='auto')
    clf_st = LinearDiscriminantAnalysis(shrinkage='auto')


    

    print('abc')

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
  #a = test_csp()
  test()