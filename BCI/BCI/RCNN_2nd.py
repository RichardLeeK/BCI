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
  seg = int((x.shape[2] - (win_size * fs)) / (upd_time * fs)) + 1
  new_x = np.zeros(shape=(x.shape[0], x.shape[1], int(win_size * fs), seg, x.shape[3]))
  for i in range(seg):
    new_x[:,:,:,i,:] = x[:,:,int(upd_time * fs * i):int(upd_time * fs * i + win_size * fs),:]
  return np.array(new_x)

def resting_filter(x, y):
  y_ = y.argmax(axis=1)
  new_x = np.zeros((len(x), int(len(x[0])/2), len(x[0][0]), len(x[0][0][0]))); new_y = [];
  k = 0
  for i in range(len(y_)):
    if y_[i] == 0:
      new_x[:,k,:,:] = x[:,i,:,:]
      new_y.append([0, 1])
      k+=1
    elif y_[i] == 1:
      new_x[:,k,:,:] = x[:,i,:,:]
      new_y.append([1, 0])
      k+=1
  return np.array(new_x), np.array(new_y)

def make_original_data(sub):
  cnt = scipy.io.loadmat('raw_KIST/twist/raw_cnt_' + sub + '.mat')['cnt'][0][0][4]
  mrk = scipy.io.loadmat('raw_KIST/twist/raw_mrk_' + sub + '.mat')['mrk'][0][0][0][0]
  y = scipy.io.loadmat('raw_KIST/twist/raw_mrk_' + sub + '.mat')['mrk'][0][0][3]
  epo = cnt_to_epo(cnt, mrk)
  y_  = np.transpose(y)
  kv = BCI.gen_kv_idx(y_, 5)
  k = 1
  for train_idx, test_idx in kv:
    train_x = epo[train_idx,:,:]
    test_x = epo[test_idx,:,:]
    train_y = y[:, train_idx]
    test_y = y[:, test_idx]
    res = {'train_x': train_x, 'test_x': test_x, 'train_y': train_y, 'test_y': test_y}
    scipy.io.savemat('RCNN2/twist_ori_nonfilter/' + sub + '_' + str(k) + '.mat', res)
    k+=1


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

def make_data_binary(sub):
  cnt = scipy.io.loadmat('raw_KIST/twist/raw_cnt_' + sub + '.mat')['cnt'][0][0][4]
  mrk = scipy.io.loadmat('raw_KIST/twist/raw_mrk_' + sub + '.mat')['mrk'][0][0][0][0]
  y = scipy.io.loadmat('raw_KIST/twist/raw_mrk_' + sub + '.mat')['mrk'][0][0][3]
  epo = []
  for i in range(1, 10):
    cnt_fs = CSP.arr_bandpass_filter(cnt, i * 4, (i + 1) * 4, 1000, 4)
    epo.append(cnt_to_epo(cnt_fs, mrk))
  y_  = np.transpose(y)
  x, y = resting_filter(np.array(epo), y_)
  kv = BCI.gen_kv_idx(y, 5)
  k = 1
  for train_idx, test_idx in kv:
    train_x = epo_temporal_dividing(np.array(epo)[:,train_idx,:,:], 3, 1)
    test_x = epo_temporal_dividing(np.array(epo)[:,test_idx,:,:], 3, 1)
    train_y = y[train_idx]
    test_y = y[test_idx]
    res = {'train_x': train_x, 'test_x': test_x, 'train_y': np.transpose(train_y), 'test_y': np.transpose(test_y)}
    scipy.io.savemat('RCNN2/twist_ori_bin_3_1/' + sub + '_' + str(k) + '.mat', res)
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

def x_translator3(x):
  xs = x.shape
  new_x = np.zeros((xs[3], xs[0], xs[1], xs[2]))
  for i in range(xs[3]):
    new_x[i] = x[:,:,:,i]
  return np.reshape(new_x, (xs[3], xs[0], xs[1], xs[2], 1))



def classification(sub):
  import RCNN
  import CNN
  for i in range(1, 6):
    train_data = scipy.io.loadmat('RCNN2/twist_rev_5/' + sub + '_' + str(i) + '_train.mat')
    test_data = scipy.io.loadmat('RCNN2/twist_rev_5/' + sub + '_' + str(i) + '_test.mat')
    train_x = np.transpose(train_data['train'][0][0][0])
    train_y = np.transpose(train_data['train'][0][0][1])

    test_x = np.transpose(test_data['test'][0][0][0])
    test_y = np.transpose(test_data['test'][0][0][1])

    #show_pca(train_x, test_x, train_y, test_y, sub + '_' + str(i))
    #model = CNN.create_model((9, 18, 1))
    train_x = tans(train_x)
    test_x = tans(test_x)
    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    model.fit(train_x, train_y.argmax(axis=1))
    score = model.score(test_x, test_y.argmax(axis=1))
    pen = open('LDA_non_filter.csv', 'a')
    #pen.write('LDA,' + sub + ',' + str(i) + ',' + str(score) + '\n')
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

def arr_tans(x):
  new_x = []
  for v in x:
    cur_x = []
    for vv in v:
      cur_x.append(vv.flatten())
    new_x.append(np.array(cur_x))
  return np.array(new_x)


def classification_binary(sub):
  for i in range(1, 6):
    train_data = scipy.io.loadmat('RCNN2/twist_rev_bin_3_1/' + sub + '_' + str(i) + '_train.mat')
    test_data = scipy.io.loadmat('RCNN2/twist_rev_bin_3_1/' + sub + '_' + str(i) + '_test.mat')
    train_x = x_translator2(train_data['train'][0][0][0])
    train_y = np.transpose(train_data['train'][0][0][1])

    test_x = x_translator2(test_data['test'][0][0][0])
    test_y = np.transpose(test_data['test'][0][0][1])

    train_x = arr_tans(train_x)
    test_x = arr_tans(test_x)
    
    for j in range(0, 9):
      model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      model.fit(train_x[:,j,:], train_y.argmax(axis=1))
      pen = open('train.csv','w');
      xxxx = train_x[:,j,:]
      
      for k in range(len(xxxx)):
        sentence = str(train_y.argmax(axis=1)[k])
        for l in range(6):
          sentence += ',' + str(xxxx[k][l])
        pen.write(sentence + '\n')
      pen.close()

      pen = open('test.csv', 'w');
      xxxx = test_x[:,j,:]
      
      for k in range(len(xxxx)):
        sentence = str(test_y.argmax(axis=1)[k])
        for l in range(6):
          sentence += ',' + str(xxxx[k][l])
        pen.write(sentence + '\n')
      pen.close()

      score = model.score(test_x[:,j,:], test_y.argmax(axis=1))
      show_pca(train_x[:,j,:], test_x[:,j,:], train_y, test_y, 'abc')
      #pen = open('LDA_seg_bin_origianl_filtering.csv', 'a')
      #pen.write('LDA,' + sub + ',' + str(i) + ',' + str(j) + ',' + str(score) + '\n')
      #pen.close()
      

def show_pca(train_x, test_x, train_y, test_y, tag):
  import matplotlib.pyplot as plt
  from sklearn.decomposition import PCA
  x_train_ = np.reshape(train_x, (len(train_x), 6))
  x_test_ =  np.reshape(test_x, (len(test_x), 6))
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
  plt.show()
  #plt.savefig('fig/seg_mv/' + tag + '.png')
  plt.close()


def classification_temporal(sub):
  import RCNN,CNN
  for i in range(1, 6):
    train_data = scipy.io.loadmat('RCNN2/twist_rev_mv/' + sub + '_' + str(i) + '_train.mat')
    test_data = scipy.io.loadmat('RCNN2/twist_rev_mv/' + sub + '_' + str(i) + '_test.mat')
    train_x = x_translator2(train_data['train'][0][0][0])
    train_y = np.transpose(train_data['train'][0][0][1])

    test_x = x_translator2(test_data['test'][0][0][0])
    test_y = np.transpose(test_data['test'][0][0][1])
    show_pca(train_x, test_x, train_y, test_y, sub + '_' + str(i))
    train_x = tans(train_x)
    test_x = tans(test_x)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(train_x, train_y.argmax(axis=1))
    train_x = lda.transform(train_x)
    test_x = lda.transform(test_x)
    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    #model = CNN.create_3d_model((9, 18, 5, 1))
    model.fit(train_x, train_y.argmax(axis=1), shuffle=True)
    score = model.score(test_x, test_y.argmax(axis=1))
    pen = open('LDA_seg_mv_res_L.csv', 'a')
    pen.write('LDA,' + sub + ',' + str(i) + ',' + str(score) + '\n')
    pen.close()
    

if __name__ == '__main__':
  for i in range(1, 14):
    make_data(str(i))
    #make_data_binary(str(i))
    #classification_binary(str(i))
    #make_data_temporal(str(i))
    #make_data_temporal(str(i))
    #make_data(str(i))
    #classification(str(i))
    #make_data_binary(str(i))