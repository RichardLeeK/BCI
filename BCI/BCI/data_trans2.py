
import scipy.io, CSP, BCI, pickle, Com_test
import numpy as np
import RCNN
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, f1_score, recall_score
def load_kist_data(sub = '3'):
  print(sub)
  x = scipy.io.loadmat('kist_data/twist/x_' + sub + '.mat')['x_' + sub].transpose()
  y = scipy.io.loadmat('kist_data/twist/y_' + sub + '.mat')['y_' + sub].transpose().argmax(axis=1)
  y_ = np.array(BCI.lab_inv_translator(y))

  kv = BCI.gen_kv_idx(y_, 5)
  k = 1
  for train_idx, test_idx in kv:
    x_train, y_train = x[train_idx], y_[train_idx]
    x_test, y_test = x[test_idx], y_[test_idx]
    file = open('kist_test_data/twist/np/' + sub + '_' + str(k) + '.pic', 'wb')
    pickle.dump({'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}, file)
    file.close()
    f_train = {'clab':[1, 2, 3], 'fs':250, 'x': [], 'y': []}
    f_test = {'clab':[1, 2, 3], 'fs':250, 'x': [], 'y': []}
    
    f_train['x'] = np.transpose(x_train)
    f_train['y'] = np.transpose(y_train)
    f_test['x'] = np.transpose(x_test)
    f_test['y'] = np.transpose(y_test)

    scipy.io.savemat('kist_test_data/twist/ori/A0' + sub + 'T_' + str(k) + '_train.mat', f_train)
    scipy.io.savemat('kist_test_data/twist/ori/A0' + sub + 'T_' + str(k) + '_test.mat', f_test)
    k += 1

def x_translator(x):
  new_x = []
  for i in range(len(x[0])):
    new_x.append(x[:,i,:])
  return np.reshape(np.array(new_x), (len(new_x), len(new_x[0]), len(new_x[0][0]), 1))

def rcnn_ts_batch(sub = '1'):
  import matplotlib.pyplot as plt
  for fold in range(1, 6):
    x_train = x_translator(scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/kist_test_data/twist/rev/A0' + sub + 'T_' + str(fold) + '_train.mat')['csp_2_train'])
    x_test = x_translator(scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/kist_test_data/twist/rev/A0' + sub + 'T_' + str(fold) + '_test.mat')['csp_2_test'])
    file = open('kist_test_data/twist/np/' + sub + '_' + str(fold) + '.pic', 'rb')
    raw = pickle.load(file)
    file.close()
    y_train = raw['y_train']
    y_test = raw['y_test']
    x_train_ = np.reshape(x_train, (len(x_train), 18 * 4))
    x_test_ = np.reshape(x_test, (len(x_test), 18 * 4))
    pca = PCA(n_components=2)
    X_r = pca.fit(x_train_).transform(x_train_)
    X2_r = pca.transform(x_test_)
    y = y_train.argmax(axis = 1)
    y2 = y_test.argmax(axis = 1)
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

    model = RCNN.create_model((18, 4, 1))
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size = 5, epochs=100)
    print('d')

def classification(sub, path):

  train = scipy.io.loadmat(path + sub + '-train.mat')
  test = scipy.io.loadmat(path + sub + '-test.mat')

  train_fold = train['total_train'][0]
  test_fold = test['total_test'][0]

  total_res_val = np.zeros((9, 21))
  total_res_val2 = np.zeros((9, 21))
  
  for i in range(6): # temporal dividing
    plt.rcParams["font.family"] = "Times New Roman"
    import seaborn as sns; sns.set()
    temporal_size = train_fold[i][0][0][0][0].shape[1]
    res_val = np.zeros((9, temporal_size))
    res_val2= np.zeros((9, temporal_size))
    for j in range(5): # 5-fold cross validation
      train_x = np.transpose(train_fold[i][0][0][0][j])
      train_y = np.transpose(train_fold[i][0][0][1])

      test_x = np.transpose(test_fold[i][0][0][0][j])
      test_y = np.transpose(test_fold[i][0][0][1])

      t_train_x = []; t_test_x = [];
      for k in range(9):
        for l in range(temporal_size):
          t_train_x.append(Com_test.arr_flatten(train_x[:,:,l,k]))
          t_test_x.append(Com_test.arr_flatten(test_x[:,:,l,k]))
      for k in range(len(t_test_x)):
        cur_train_x = t_train_x[k]
        cur_test_x = t_test_x[k]
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(cur_train_x, train_y.argmax(axis=1))
        y_predict = lda.predict(cur_test_x)
        coh = cohen_kappa_score(test_y.argmax(axis=1), y_predict)
        acc = lda.score(cur_test_x, test_y.argmax(axis=1))
        y_val = k % temporal_size
        x_val = int(k / temporal_size)
        res_val[x_val, y_val] += coh
        res_val2[x_val, y_val] += acc
        total_res_val[x_val, int(-0.5*i*i + 6.5*i + y_val)] += coh
        total_res_val2[x_val, int(-0.5*i*i + 6.5*i + y_val)] += acc
    res_val /= 5
    res_val2 /= 5
  total_res_val /= 5
  total_res_val2 /= 5
  plt.rcParams["font.family"] = "Times New Roman"
  plt.figure(figsize=(12, 4))
  ax = sns.heatmap(total_res_val, cmap="BuGn", vmin=0.1, vmax=0.85, square=True, annot=True)
  plt.savefig('KIST/grasp/' + sub + 'coh.png', format='png', dpi=1000)
  plt.close()
  plt.figure(figsize=(12, 4))
  ax = sns.heatmap(total_res_val2, cmap="BuGn", vmin=0.1, vmax=0.85, square=True, annot=True)
  plt.savefig('KIST/grasp/' + sub + 'acc.png', format='png', dpi=1000)
  plt.close()

  pen = open('KIST/grasp/res.csv', 'a')
  pen.write(sub + ',' + str(total_res_val.max()) + ',' + str(total_res_val2.max()) + '\n')
  pen.close()



  print('abc')

def classification_1105(sub, path):
  # new version (merging)
  import seaborn as sns; sns.set()
  train = scipy.io.loadmat(path + sub + '-train.mat')
  test = scipy.io.loadmat(path + sub + '-test.mat')

  train_fold = train['total_train'][0]
  test_fold = test['total_test'][0]

  total_res_val = np.zeros((9, 21))
  total_res_val2 = np.zeros((9, 21))
  txs = []; tys= []; vxs = []; vys = [];
  for i in range(6): # temporal dividing    
    temporal_size = train_fold[i][0][0][0][0].shape[1]
    res_val = np.zeros((9, temporal_size))
    res_val2= np.zeros((9, temporal_size))
    for j in range(5): # 5-fold cross validation
      train_x = np.transpose(train_fold[i][0][0][0][j])
      train_y = np.transpose(train_fold[i][0][0][1])

      test_x = np.transpose(test_fold[i][0][0][0][j])
      test_y = np.transpose(test_fold[i][0][0][1])

      t_train_x = []; t_test_x = [];
      for k in range(9):
        for l in range(temporal_size):
          
          t_train_x.append(Com_test.arr_flatten(train_x[:,:,l,k]))
          t_test_x.append(Com_test.arr_flatten(test_x[:,:,l,k]))
      for k in range(len(t_test_x)):
        cur_train_x = t_train_x[k]
        cur_test_x = t_test_x[k]
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(cur_train_x, train_y.argmax(axis=1))
        y_predict = lda.predict(cur_test_x)
        coh = cohen_kappa_score(test_y.argmax(axis=1), y_predict)
        acc = lda.score(cur_test_x, test_y.argmax(axis=1))
        y_val = k % temporal_size
        x_val = int(k / temporal_size)
        res_val[x_val, y_val] += coh
        res_val2[x_val, y_val] += acc
        total_res_val[x_val, int(-0.5*i*i + 6.5*i + y_val)] += coh
        total_res_val2[x_val, int(-0.5*i*i + 6.5*i + y_val)] += acc
    res_val /= 5
    res_val2 /= 5
  total_res_val /= 5
  total_res_val2 /= 5
  plt.rcParams["font.family"] = "Times New Roman"
  plt.figure(figsize=(12, 4))
  ax = sns.heatmap(total_res_val, cmap="BuGn", vmin=0.1, vmax=0.85, square=True, annot=True)
  plt.savefig('KIST/grasp/' + sub + 'coh.png', format='png', dpi=1000)
  plt.close()
  plt.figure(figsize=(12, 4))
  ax = sns.heatmap(total_res_val2, cmap="BuGn", vmin=0.1, vmax=0.85, square=True, annot=True)
  plt.savefig('KIST/grasp/' + sub + 'acc.png', format='eps', dpi=1000)
  plt.close()

#  pen = open('KIST/grasp/res.csv', 'a')
#  pen.write(sub + ',' + str(total_res_val.max()) + ',' + str(total_res_val2.max()) + '\n')
#  pen.close()



  print('abc')



def classification_tdp(sub, path):
  import seaborn as sns; sns.set()
  train = scipy.io.loadmat(path + sub + '-train.mat')
  test = scipy.io.loadmat(path + sub + '-test.mat')

  train_fold = train['ttotal_tdp_train'][0][0]
  test_fold = test['ttotal_tdp_test'][0][0]
  temporal_size = 7
  total_res_val = np.zeros((15, 7))
  total_res_val2 = np.zeros((15, 7))
  for j in range(5): # 5-fold cross validation
    train_x = np.transpose(train_fold[0][j])
    train_y = np.transpose(train_fold[1])

    test_x = np.transpose(test_fold[0][j])
    test_y = np.transpose(test_fold[1])

    t_train_x = []; t_test_x = [];
    for k in range(temporal_size):
      for l in range(7):
        t_train_x.append(Com_test.arr_flatten(train_x[:,:,l,k]))
        t_test_x.append(Com_test.arr_flatten(test_x[:,:,l,k]))
    for k in range(len(t_test_x)):
      cur_train_x = t_train_x[k]
      cur_test_x = t_test_x[k]
      lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      lda.fit(cur_train_x, train_y.argmax(axis=1))
      y_predict = lda.predict(cur_test_x)
      coh = cohen_kappa_score(test_y.argmax(axis=1), y_predict)
      acc = lda.score(cur_test_x, test_y.argmax(axis=1))
      x_val = k % temporal_size
      y_val = int(k / temporal_size)

      total_res_val[x_val, y_val] += coh
      total_res_val2[x_val, y_val] += acc
  total_res_val /= 5
  total_res_val2 /= 5
  plt.rcParams["font.family"] = "Times New Roman"
  plt.figure(figsize=(12, 4))
  ax = sns.heatmap(total_res_val, cmap="BuGn", vmin=0.1, vmax=0.85, square=True, annot=True)
  plt.savefig('KIST/twist_tdp/' + sub + 'coh.png', format='png', dpi=1000)
  plt.close()
  plt.figure(figsize=(12, 4))
  ax = sns.heatmap(total_res_val2, cmap="BuGn", vmin=0.1, vmax=0.85, square=True, annot=True)
  plt.savefig('KIST/twist_tdp/' + sub + 'acc.png', format='png', dpi=1000)
  plt.close()

  pen = open('KIST/twist_tdp/res.csv', 'a')
  pen.write(sub + ',' + str(total_res_val.max()) + ',' + str(total_res_val2.max()) + '\n')
  pen.close()



def batch_classification_access():
  import seaborn as sns; sns.set()
  path = 'F:/KIST/source/BCI/BCI/BCI/Access/no_trick/'
  for i in range(1, 10):
    res1 = np.zeros((21, 9))
    res2 = np.zeros((21, 9))
    for j in range(1, 5):
      train = scipy.io.loadmat(path + str(i) + '_' + str(j) + '_train.mat')['total_train'][0][0]
      test = scipy.io.loadmat(path + str(i) + '_' + str(j) + '_test.mat')['total_test'][0][0]
      train_x  = np.transpose(train[0])
      train_y = np.transpose(train[2])
      test_x = np.transpose(test[0])
      test_y = np.transpose(test[2])
      idx = np.transpose(train[1])
      subject = train[3][0]

      for k in range(train_x.shape[2]):
        cur_tx = train_x[:,:,k] # current train x
        cur_vx = test_x[:,:,k] # current validation x
        cur_w = idx[:,k]
        cur_stx = cur_tx[:,cur_w.argmax()] # current selected train x
        cur_stx = cur_stx.reshape(len(cur_stx), 1)
        cur_svx = cur_vx[:,cur_w.argmax()]
        cur_svx = cur_svx.reshape(len(cur_svx), 1)

        accs = []; cohs = [];
        for l in range(1, 18):
          print(str(i) + '_' + str(j) + '_' + str(k) + '_' + str(l))
          cur_nstx = np.squeeze(np.array(cur_tx[:,np.where(cur_w==cur_w.max()-l)]))
          cur_nsvx = np.squeeze(np.array(cur_vx[:,np.where(cur_w==cur_w.max()-l)]))

          cur_nstx = cur_nstx.reshape(len(cur_nstx), 1)
          cur_nsvx = cur_nsvx.reshape(len(cur_nsvx), 1)
          cur_stx = np.concatenate((cur_stx, cur_nstx), 1)
          cur_svx = np.concatenate((cur_svx, cur_nsvx), 1)
          lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
          lda.fit(cur_stx, train_y.argmax(axis=1))
          y_predict = lda.predict(cur_svx)
          coh = cohen_kappa_score(test_y.argmax(axis=1), y_predict)
          acc = lda.score(cur_svx, test_y.argmax(axis=1))
          accs.append(acc); cohs.append(coh);


        x_val = int(k / 9)
        y_val = k % 9

        res1[x_val, y_val] += np.array(accs).max()
        res2[x_val, y_val] += np.array(cohs).max()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(5, 10))
    ax = sns.heatmap(res1/5, cmap="BuGn", vmin=0.1, vmax=0.85, square=True, annot=True)
    plt.savefig('Access/f_no_trick2/' + subject + '_acc.png', format='png', dpi=1000)
    plt.close()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(5, 10))
    ax = sns.heatmap(res2/5, cmap="BuGn", vmin=0.1, vmax=0.85, square=True, annot=True)
    plt.savefig('Access/f_no_trick2/' + subject + '_coh.png', format='png', dpi=1000)
    plt.close()
    
def comp_to_mat(sub):
  import Competition, Com_test
  path = 'data/A0'+ sub + 'T.npz'
  cnt, mrk, dur, y = Competition.load_cnt_mrk_y(path)
  epo = Com_test.cnt_to_epo(cnt, mrk, dur)
  epo, y = Com_test.out_label_remover(epo, y)
  y = BCI.lab_inv_translator(y, 4)
  res = {'epo': np.transpose(epo), 'y': np.transpose(y)}
  scipy.io.savemat('Access/com/'+sub+'.mat', res)
  print("abc")

if __name__ == '__main__':
  path = 'E:/Richard/TSOSPData/grasp/'
#  files_raw = os.listdir(path)
#  files = []
#  for f in files_raw:
#    files.append(f.split('-')[0])
#  files = list(set(files))

#  for f in files:
#    classification_1105(f, path)
  classification_1105('20180615_jgyoon3_grasp_MI.mat', path)


#  for i in range(1, 10):
#    comp_to_mat(str(i))
    

  