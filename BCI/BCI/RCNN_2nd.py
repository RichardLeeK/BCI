import CSP, scipy.io, BCI
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
def cnt_to_epo(cnt, mrk, ival=3000):
  epo = []
  for v in mrk:
    epo.append(np.array(cnt[v : v + ival]))
  epo = np.array(epo)
  return epo

def make_data(sub):
  cnt = scipy.io.loadmat('raw_KIST/twist/raw_cnt_' + sub + '.mat')['cnt'][0][0][4]
  mrk = scipy.io.loadmat('raw_KIST/twist/raw_mrk_' + sub + '.mat')['mrk'][0][0][0][0]
  y = scipy.io.loadmat('raw_KIST/twist/raw_mrk_' + sub + '.mat')['mrk'][0][0][3]
  epo = []
  for i in range(1, 10):
    cnt_fs = CSP.arr_bandpass_filter(cnt, i * 4, (i + 1) * 4, 1000, 4)
    epo.append(cnt_to_epo(cnt_fs, mrk))

  y_  = np.transpose(y)
  kv = BCI.gen_kv_idx(y_, 5)
  k = 1
  for train_idx, test_idx in kv:
    train_x = np.array(epo)[:,train_idx,:,:]
    test_x = np.array(epo)[:,test_idx,:,:]
    train_y = y[:, train_idx]
    test_y = y[:, test_idx]
    res = {'train_x': train_x, 'test_x': test_x, 'train_y': train_y, 'test_y': test_y}
    scipy.io.savemat('RCNN2/twist_ori/' + sub + '_' + str(k) + '.mat', res)
    k+=1

def epo_temporal_dividing(x, win_size, upd_time, fs = 1000):
  seg = (x.shape[2] - (win_size * fs)) / (upd_time * fs)
  new_x = np.zeros(shape=(x.shape[0], x.shape[1], int(win_size * fs), seg, x.shape[3]))
  for i in range(seg):
    new_x[:,:,:,i,:] = x[:,:,int(upd_time * fs * i):int(upd_time * fs * i + win_size * fs),:]
  return np.array(new_x)

def make_data_temporal(sub):
  cnt = scipy.io.loadmat('raw_KIST/twist/raw_cnt_' + sub + '.mat')['cnt'][0][0][4]
  mrk = scipy.io.loadmat('raw_KIST/twist/raw_mrk_' + sub + '.mat')['mrk'][0][0][0][0]
  y = scipy.io.loadmat('raw_KIST/twist/raw_mrk_' + sub + '.mat')['mrk'][0][0][3]
  epo = []
  for i in range(1, 10):
    cnt_fs = CSP.arr_bandpass_filter(cnt, i * 4, (i + 1) * 4, 1000, 4)
    epo.append(cnt_to_epo(cnt_fs, mrk))
  y_  = np.transpose(y)
  kv = BCI.gen_kv_idx(y_, 5)
  k = 1

  for train_idx, test_idx in kv:
    train_x = epo_temporal_dividing(np.array(epo)[:,train_idx,:,:], 0.6, 0.3)
    test_x = epo_temporal_dividing(np.array(epo)[:,test_idx,:,:], 0.6, 0.3)
    train_y = y[:, train_idx]
    test_y = y[:, test_idx]
    res = {'train_x': train_x, 'test_x': test_x, 'train_y': train_y, 'test_y': test_y}
    scipy.io.savemat('RCNN2/twist_ori_dj/' + sub + '_' + str(k) + '.mat', res)
    k+=1

def make_data_moving(sub):
  segment_idx = 5
  cnt = scipy.io.loadmat('raw_KIST/twist/raw_cnt_' + sub + '.mat')['cnt'][0][0][4]
  mrk = scipy.io.loadmat('raw_KIST/twist/raw_mrk_' + sub + '.mat')['mrk'][0][0][0][0]
  y = scipy.io.loadmat('raw_KIST/twist/raw_mrk_' + sub + '.mat')['mrk'][0][0][3]
  epo = []
  for i in range(1, 10):
    cnt_fs = CSP.arr_bandpass_filter(cnt, i * 4, (i + 1) * 4, 1000, 4)
    epo.append(cnt_to_epo(cnt_fs, mrk))
  y_  = np.transpose(y)
  kv = BCI.gen_kv_idx(y_, 5)
  k = 1

  for train_idx, test_idx in kv:
    train_x = epo_temporal_dividing(np.array(epo)[:,train_idx,:,:], segment_idx)
    test_x = epo_temporal_dividing(np.array(epo)[:,test_idx,:,:], segment_idx)
    train_y = y[:, train_idx]
    test_y = y[:, test_idx]
    res = {'train_x': train_x, 'test_x': test_x, 'train_y': train_y, 'test_y': test_y}
    scipy.io.savemat('RCNN2/twist_ori_5/' + sub + '_' + str(k) + '.mat', res)
    k+=1


def x_translator(x):
  new_x = np.zeros((len(x[0][0]), len(x), len(x[0])))
  for i in range(len(x[0][0])):
    new_x[i] = x[:,:,i]
  return np.reshape(np.array(new_x), (len(x[0][0]), len(x), len(x[0]), 1))
  
def x_translator2(x):
  new_x = np.zeros((len(x[0][0][0]), len(x), len(x[0]), len(x[0][0])))
  for i in range(len(x[0][0][0])):
    new_x[i] = x[:,:,:,i]
  return np.reshape(np.array(new_x), (len(x[0][0][0]), len(x), len(x[0]), len(x[0][0]), 1))

def classification(sub):
  import RCNN
  import CNN
  for i in range(1, 6):
    train_data = scipy.io.loadmat('RCNN2/twist_rev/' + sub + '_' + str(i) + '_train.mat')
    test_data = scipy.io.loadmat('RCNN2/twist_rev/' + sub + '_' + str(i) + '_test.mat')
    train_x = x_translator(train_data['train'][0][0][0])
    train_y = np.transpose(train_data['train'][0][0][1])

    test_x = x_translator(test_data['test'][0][0][0])
    test_y = np.transpose(test_data['test'][0][0][1])

    show_pca(train_x, test_x, train_y, test_y, sub + '_' + str(i))
    #model = CNN.create_model((9, 18, 1))
    train_x = tans(train_x)
    test_x = tans(test_x)
    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    model.fit(train_x, train_y.argmax(axis=1))
    score = model.score(test_x, test_y.argmax(axis=1))
    pen = open('LDA_seg_1_res_L.csv', 'a')
    pen.write('LDA,' + sub + ',' + str(i) + ',' + str(score) + '\n')
    pen.close()
    
def make_binary(x, y):
  y_ = y.argmax(axis=1)
  new_y = []
  new_x = []
  for v in range(len(y_)):
    if y_[v] == 1:
      new_y.append([0, 1])
      new_x.append(x[v])
    elif y_[v] == 2:
      new_y.append([1, 0])
      new_x.append(x[v])
  return np.array(new_x), np.array(new_y)

def tans(x):
  new_x = []
  for v in x:
    new_x.append(v.flatten())
  return np.array(new_x)

def classification_binary(sub):
  import RCNN,CNN
  for i in range(1, 6):
    train_data = scipy.io.loadmat('RCNN2/twist_rev_5/' + sub + '_' + str(i) + '_train.mat')
    test_data = scipy.io.loadmat('RCNN2/twist_rev_5/' + sub + '_' + str(i) + '_test.mat')
    train_x = x_translator2(train_data['train'][0][0][0])
    train_y = np.transpose(train_data['train'][0][0][1])
    train_x, train_y = make_binary(train_x, train_y)
    test_x = x_translator2(test_data['test'][0][0][0])
    test_y = np.transpose(test_data['test'][0][0][1])
    test_x, test_y = make_binary(test_x, test_y)
    model = CNN.create_3d_model((9, 18, 5, 1))
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100, batch_size=10)
    score = model.evaluate(test_x, test_y)

def show_pca(train_x, test_x, train_y, test_y, tag):
  import matplotlib.pyplot as plt
  from sklearn.decomposition import PCA
  x_train_ = np.reshape(train_x, (len(train_x), 9*3*18))
  x_test_ =  np.reshape(test_x, (len(test_x), 9*3*18))
  pca = PCA(n_components=2)
  X_r = pca.fit(x_train_).transform(x_train_)
  X2_r = pca.transform(x_test_)
  y = train_y.argmax(axis = 1)
  y2 = test_y.argmax(axis = 1)
  plt.figure()
  colors = ['navy', 'turquoise', 'darkorange']
  colors2 = ['black', 'green', 'yellow']
  target_names = ['1', '2', '3']
  for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=2,
              label=target_name)
  for color, i, target_name in zip(colors2, [0, 1, 2], target_names):
    plt.scatter(X2_r[y2 == i, 0], X2_r[y2 == i, 1], color=color, alpha=.8, lw=2,
              label=target_name)

  plt.savefig('fig/seg_3/' + tag + '.png')
  plt.close()


def classification_temporal(sub):
  import RCNN,CNN
  for i in range(1, 6):
    train_data = scipy.io.loadmat('RCNN2/twist_rev_3/' + sub + '_' + str(i) + '_train.mat')
    test_data = scipy.io.loadmat('RCNN2/twist_rev_3/' + sub + '_' + str(i) + '_test.mat')
    train_x = x_translator2(train_data['train'][0][0][0])
    train_y = np.transpose(train_data['train'][0][0][1])

    test_x = x_translator2(test_data['test'][0][0][0])
    test_y = np.transpose(test_data['test'][0][0][1])
    show_pca(train_x, test_x, train_y, test_y, sub + '_' + str(i))
    train_x = tans(train_x)
    test_x = tans(test_x)

    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    #model = CNN.create_3d_model((9, 18, 5, 1))
    model.fit(train_x, train_y.argmax(axis=1))
    score = model.score(test_x, test_y.argmax(axis=1))
    pen = open('LDA_seg_3_res_L.csv', 'a')
    pen.write('LDA,' + sub + ',' + str(i) + ',' + str(score) + '\n')
    pen.close()
    

if __name__ == '__main__':
  for i in range(1, 14):
    #classification_binary(str(i))
    classification_temporal(str(i))
    #make_data_temporal(str(i))
    #make_data(str(i))
    #classification(str(i))