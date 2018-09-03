import CSP, scipy.io, BCI

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

def epo_temporal_dividing(x, seg):
  new_x = np.zeros(shape=(x.shape[0], x.shape[1], int(x.shape[2]/seg), seg, x.shape[3]))
  for i in range(seg):
    new_x[:,:,:,i,:] = x[:,:,int(x.shape[2]/seg) * i:int(x.shape[2]/seg) * (i + 1),:]
  return np.array(new_x)

def make_data_temporal(sub):
  segment_idx = 10
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
    scipy.io.savemat('RCNN2/twist_temp_10/' + sub + '_' + str(k) + '.mat', res)
    k+=1

def x_translator(x):
  new_x = np.zeros((len(x[0][0]), len(x), len(x[0])))
  for i in range(len(x[0][0])):
    new_x[i] = x[:,:,i]
  return np.reshape(np.array(new_x), (len(x[0][0]), len(x), len(x[0]), 1))
  


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

    model = CNN.create_model((9, 18, 1))
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=10)
    score = model.evaluate(test_x, test_y)
    pen = open('cnn2_res.csv', 'a')
    pen.write('CNN,' + sub + ',' + str(i) + ',' + str(score[1]) + '\n')
    pen.close()
    

if __name__ == '__main__':
  for i in range(1, 14):
    make_data_temporal(str(i))
    make_data(str(i))
    #classification(str(i))