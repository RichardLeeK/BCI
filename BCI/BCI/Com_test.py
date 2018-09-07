import Competition, BCI, scipy.io, CSP, MIBIF
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, f1_score, recall_score

def new_numpy_to_mat(sub = '1'):
  path='data/A0' + sub + 'T.npz'
  x, y = Competition.load_one_data(path)
  x = CSP.bandpass_filter(x, 5, 30, 250)
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
  for v in file:
    v['x'] = np.transpose(np.array(v['x']))
    v['y'] = np.transpose(np.array(v['y']))

  scipy.io.savemat('mat/ori_original/' + path.split('/')[-1].split('.')[0] + '_1.mat', file[0])
  scipy.io.savemat('mat/ori_original/' + path.split('/')[-1].split('.')[0] + '_2.mat', file[1])
  scipy.io.savemat('mat/ori_original/' + path.split('/')[-1].split('.')[0] + '_3.mat', file[2])


def load_new_mat(sub = '1'):
  
  aa = scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/res_original/A0' + sub + 'T_CSP.mat')
  x, y = Competition.load_one_data('data/A0' + sub + 'T.npz')
  y = np.array(BCI.lab_inv_translator(y, 4))
  x_ = np.transpose(aa['csp_res'])
  kv = BCI.gen_kv_idx(y, 10)
  accs = []; pres = []; recs = []; f1s = []; cohens = [];
  for train_idx, test_idx in kv:
    x_train, y_train = x_[train_idx], y[train_idx]
    x_test, y_test = x_[test_idx], y[test_idx]
    from sklearn.svm import SVC
    clf2 = SVC(kernel='rbf') 
    
    #clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    clf2.fit(x_train, y_train.argmax(axis=1))
    y_predict = clf2.predict(x_test)

    cohens.append(cohen_kappa_score(y_predict, y_test.argmax(axis=1)))

    accs.append(accuracy_score(y_test.argmax(axis=1), y_predict))
    pres.append(precision_score(y_test.argmax(axis=1), y_predict, average='macro'))
    recs.append(recall_score(y_test.argmax(axis=1), y_predict, average='macro'))
    f1s.append(f1_score(y_test.argmax(axis=1), y_predict, average='macro'))


    # PRE, REC, ACC, F1, KAP
  pen = open("CSP_SVM.csv", 'a')
  pen.write(sub + ',' + str(np.mean(accs)) + ',' + str(np.mean(pres)) + ',' + str(np.mean(recs)) + ',' + str(np.mean(f1s)) + ',' + str(np.mean(cohens)) + '\n')

  print('abc')

def one_fbcsp(x, y):
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
  for v in file:
    v['x'] = np.transpose(np.array(v['x']))
    v['y'] = np.transpose(np.array(v['y']))
  return file

def gen_ts_topo(sub = '1'):
  optimal_bank = [20,30,19,9,9,20,20,19,22]
  path = 'data/A0'+ sub + 'T.npz'
  x, y = Competition.load_one_data(path)
  x = np.array(x)
  step = int(len(x[0][0]) / 5)
  for i in range(0, 5):
    cur_x = x[:, :, step * i:step * (i + 1)]
    for j in range(1, 10):
      optimal_idx = optimal_bank[int(sub) - 1]
      pred_j = optimal_idx % 9
      pred_i = int(optimal_idx / 9)
      if pred_i == i and pred_j == j - 1:
        lab_1 = {'clab':['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'], 'fs':250, 'x': [], 'y': []}
        lab_2 = {'clab':['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'], 'fs':250, 'x': [], 'y': []}
        lab_3 = {'clab':['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'], 'fs':250, 'x': [], 'y': []}
        lab_4 = {'clab':['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'], 'fs':250, 'x': [], 'y': []}
        cur_cur_x = CSP.arr_bandpass_filter(cur_x, j*4, (j+1)*4, 250)
        for k in range(len(y)):
          if y[k] == 0:
            lab_1['x'].append(np.transpose(cur_cur_x[k]))
            lab_1['y'].append([1, 0, 0, 0])
          elif y[k] == 1:
            lab_2['x'].append(np.transpose(cur_cur_x[k]))
            lab_2['y'].append([0, 1, 0, 0])
          elif y[k] == 2:
            lab_3['x'].append(np.transpose(cur_cur_x[k]))
            lab_3['y'].append([0, 0, 1, 0])
          elif y[k] == 3:
            lab_4['x'].append(np.transpose(cur_cur_x[i]))
            lab_4['y'].append([0, 0, 0, 1])
        scipy.io.savemat('mat/topo_ori/A0' + sub + 'T_1.mat', lab_1)
        scipy.io.savemat('mat/topo_ori/A0' + sub + 'T_2.mat', lab_2)
        scipy.io.savemat('mat/topo_ori/A0' + sub + 'T_3.mat', lab_3)
        scipy.io.savemat('mat/topo_ori/A0' + sub + 'T_4.mat', lab_4)



        print('abc')



def gen_fbcsp(path='data/A01T.npz'):
  x, y = Competition.load_one_data(path)
  for i in range(1, 10):
    print(i)
    x_ = CSP.arr_bandpass_filter(x, i*4, (i+1)*4, 250)
    f = one_fbcsp(x_, y)
    scipy.io.savemat('mat/fbcsp_ori/A01T_' + str(i) + '_1.mat', f[0])
    scipy.io.savemat('mat/fbcsp_ori/A01T_' + str(i) + '_2.mat', f[1])
    scipy.io.savemat('mat/fbcsp_ori/A01T_' + str(i) + '_3.mat', f[2])


def cnt_to_epo(cnt, mrk, dur):
  epo = []
  for i in range(len(mrk)):
    epo.append(np.array(cnt[mrk[i] : mrk[i] + dur[i]]))
  epo = np.array(epo)
  return epo

def out_label_remover(x, y):
  new_x = []; new_y = [];
  for i in range(len(y)):
    if y[i] == 4 or y[i] == 5:
      None
    else:
      new_x.append(np.array(x[i]))
      new_y.append(int(y[i]))
  return np.array(new_x), np.array(new_y)

def epo_temporal_dividing(x, win_size, upd_time, fs = 250):
  seg = int((x.shape[2] - (win_size * fs)) / (upd_time * fs)) + 1
  new_x = np.zeros(shape=(x.shape[0], x.shape[1], int(win_size * fs), seg, x.shape[3]))
  for i in range(seg):
    new_x[:,:,:,i,:] = x[:,:,int(upd_time * fs * i):int(upd_time * fs * i + win_size * fs),:]
  return np.array(new_x)

def gen_tscsp(sub='1'):
  path = 'data/A0'+ sub + 'T.npz'
  cnt, mrk, dur, y = Competition.load_cnt_mrk_y(path)
  epo = []
  for i in range(1, 10): # band-pass filtering
    cnt_fs = CSP.arr_bandpass_filter(cnt, i * 4, (i + 1) * 4, 250, 4)
    cur_epo = cnt_to_epo(cnt_fs, mrk, dur)
    cur_epo, y_ = out_label_remover(cur_epo, y)
    epo.append(cur_epo)
  y = BCI.lab_inv_translator(y_, 4)
  kv = BCI.gen_kv_idx(y, 5)
  k = 1
  for train_idx, test_idx in kv:
    train_x = epo_temporal_dividing(np.array(epo)[:,train_idx,:,:], 4, 0.5, 250)
    test_x = epo_temporal_dividing(np.array(epo)[:,test_idx,:,:], 4, 0.5, 250)
    train_y = np.transpose(np.array(y)[train_idx])
    test_y = np.transpose(np.array(y)[test_idx])
    res = {'train_x': train_x, 'test_x': test_x, 'train_y': train_y, 'test_y': test_y}
    scipy.io.savemat('competition/ori_4_0.5/' + sub + '_' + str(k) + '.mat', res)
    k += 1

def trick_ori(train_x, test_x, train_y, test_y):
  cls = {'clab': [], 'fs': 250, 'tx':train_x, 'ty':[[],[],[]], 'vx':test_x, 'vy':[[],[],[]]}
  cls['clab'] = [['01','23'],['02','13'],['03','12']]

  for i in range(len(train_y)):
    if train_y[i] == 0:
      cls['ty'][0].append([1, 0])
      cls['ty'][1].append([1, 0])
      cls['ty'][2].append([1, 0])
    elif train_y[i] == 1:
      cls['ty'][0].append([1, 0])
      cls['ty'][1].append([0, 1])
      cls['ty'][2].append([0, 1])
    elif train_y[i] == 2:      
      cls['ty'][0].append([0, 1])
      cls['ty'][1].append([1, 0])
      cls['ty'][2].append([0, 1])
    elif train_y[i] == 3:
      cls['ty'][0].append([0, 1])
      cls['ty'][1].append([0, 1])
      cls['ty'][2].append([1, 0])
  
  for i in range(len(test_y)):
    if test_y[i] == 0:
      cls['vy'][0].append([1, 0])
      cls['vy'][1].append([1, 0])
      cls['vy'][2].append([1, 0])
    elif test_y[i] == 1:
      cls['vy'][0].append([1, 0])
      cls['vy'][1].append([0, 1])
      cls['vy'][2].append([0, 1])
    elif test_y[i] == 2:      
      cls['vy'][0].append([0, 1])
      cls['vy'][1].append([1, 0])
      cls['vy'][2].append([0, 1])
    elif test_y[i] == 3:
      cls['vy'][0].append([0, 1])
      cls['vy'][1].append([0, 1])
      cls['vy'][2].append([1, 0])

  return cls

def trick_dac(train_x, test_x, train_y, test_y):
  # divide and conquer
  cls = {'clab': [], 'fs': 250, 'tx':[[],[],[]], 'ty':[[],[],[]], 'vx':[[],[],[]], 'vy':[[],[],[]]}
  cls['clab'] = [['01','23'],['02','13'],['03','12']]
  for i in range(len(train_y)):
    if train_y[i] == 0:
      cls['tx'][0].append(train_x[i])
      cls['ty'][0].append([1, 0])
      cls['tx'][1].append(train_x[i])
      cls['ty'][1].append([1, 0])
      cls['tx'][1].append(train_x[i])
      cls['ty'][1].append([1, 0]) 
  

  return 1

def trick_pw():
  # pair wise
  return 1

def trick_ovr():
  # one versus rest
  cls = {'clab': [], 'fs': 250, 'tx':[[],[],[]], 'ty':[[],[],[]], 'vx':[[],[],[]], 'vy':[[],[],[]]}
  cls['clab'] = [['0','123'],['1','13'],['03','12']]
  for i in range(len(train_y)):
    if train_y[i] == 0:
      cls['tx'][0].append(train_x[i])
      cls['ty'][0].append([1, 0])
      cls['tx'][1].append(train_x[i])
      cls['ty'][1].append([1, 0])
      cls['tx'][2].append(train_x[i])
      cls['ty'][2].append([1, 0])
    elif train_y[i] == 1:
      cls['tx'][0].append(train_x[i])
      cls['ty'][0].append([1, 0])
      cls['tx'][1].append(train_x[i])
      cls['ty'][1].append([0, 1])
      cls['tx'][2].append(train_x[i])
      cls['ty'][2].append([0, 1])
    elif train_y[i] == 2:      
      cls['tx'][0].append(train_x[i])
      cls['ty'][0].append([0, 1])
      cls['tx'][1].append(train_x[i])
      cls['ty'][1].append([1, 0])
      cls['tx'][2].append(train_x[i])
      cls['ty'][2].append([0, 1])
    elif train_y[i] == 3:
      cls['tx'][0].append(train_x[i])
      cls['ty'][0].append([0, 1])
      cls['tx'][1].append(train_x[i])
      cls['ty'][1].append([0, 1])
      cls['tx'][2].append(train_x[i])
      cls['ty'][2].append([1, 0])
  
  for i in range(len(test_y)):
    if test_y[i] == 0:
      cls['vx'][0].append(test_x[i])
      cls['vy'][0].append([1, 0])
      cls['vx'][1].append(test_x[i])
      cls['vy'][1].append([1, 0])
      cls['vx'][2].append(test_x[i])
      cls['vy'][2].append([1, 0])
    elif test_y[i] == 1:
      cls['vx'][0].append(test_x[i])
      cls['vy'][0].append([1, 0])
      cls['vx'][1].append(test_x[i])
      cls['vy'][1].append([0, 1])
      cls['vx'][2].append(test_x[i])
      cls['vy'][2].append([0, 1])
    elif test_y[i] == 2:      
      cls['vx'][0].append(test_x[i])
      cls['vy'][0].append([0, 1])
      cls['vx'][1].append(test_x[i])
      cls['vy'][1].append([1, 0])
      cls['vx'][2].append(test_x[i])
      cls['vy'][2].append([0, 1])
    elif test_y[i] == 3:
      cls['vx'][0].append(test_x[i])
      cls['vy'][0].append([0, 1])
      cls['vx'][1].append(test_x[i])
      cls['vy'][1].append([0, 1])
      cls['vx'][2].append(test_x[i])
      cls['vy'][2].append([1, 0])

  return cls


  return 1



def gen_tscsp_trick(sub):
  path = 'data/A0'+ sub + 'T.npz'
  cnt, mrk, dur, y = Competition.load_cnt_mrk_y(path)
  epo = []
  for i in range(1, 10): # band-pass filtering
    cnt_fs = CSP.arr_bandpass_filter(cnt, i * 4, (i + 1) * 4, 250, 4)
    cur_epo = cnt_to_epo(cnt_fs, mrk, dur)
    cur_epo, y_ = out_label_remover(cur_epo, y)
    epo.append(cur_epo)
  y = BCI.lab_inv_translator(y_, 4)
  kv = BCI.gen_kv_idx(y, 5)
  k = 1
  for train_idx, test_idx in kv:
    train_x = epo_temporal_dividing(np.array(epo)[:,train_idx,:,:], 3.5, 0.5, 250)
    test_x = epo_temporal_dividing(np.array(epo)[:,test_idx,:,:], 3.5, 0.5, 250)
    cls = trick_ori(train_x, test_x, np.array(y)[train_idx].argmax(axis=1), np.array(y)[test_idx].argmax(axis=1)) 
    scipy.io.savemat('competition/trick/ori_3.5_0.5/' + sub + '_' + str(k) + '.mat', {'cls': cls})
    k += 1

def arr_flatten(x):
  new_x = []
  for v in x:
    new_x.append(v.flatten())
  return np.array(new_x)


def classification(sub):
  temporal_size = 7
  import matplotlib.pyplot as plt
  plt.rcParams["font.family"] = "Times New Roman"
  import seaborn as sns; sns.set()
  res_val = np.zeros((9, temporal_size))

  for i in range(1, 6):
    train_data = scipy.io.loadmat('competition/rev_4.5_0.5/' + sub + '_' + str(i) + '_train.mat')
    test_data = scipy.io.loadmat('competition/rev_4.5_0.5/' + sub + '_' + str(i) + '_test.mat')
    train_x = np.transpose(train_data['train'][0][0][0])
    train_y = np.transpose(train_data['train'][0][0][1])
    test_x = np.transpose(test_data['test'][0][0][0])
    test_y = np.transpose(test_data['test'][0][0][1])

    t_train_x = []; t_test_x = [];
    for k in range(0, 9):
      for j in range(0, temporal_size):
        t_train_x.append(arr_flatten(train_x[:,j,:,k]))
        t_test_x.append(arr_flatten(test_x[:,j,:,k]))
    for j in range(len(t_test_x)):
      cur_train_x = t_train_x[j]
      cur_test_x = t_test_x[j]
      lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      lda.fit(cur_train_x, train_y.argmax(axis=1))
      y_predict = lda.predict(cur_test_x)
      coh = cohen_kappa_score(test_y.argmax(axis=1), y_predict)
      acc = accuracy_score(test_y.argmax(axis=1), y_predict)
      pre = precision_score(test_y.argmax(axis=1), y_predict, average='macro')
      rec = recall_score(test_y.argmax(axis=1), y_predict, average='macro')
      f1 = f1_score(test_y.argmax(axis=1), y_predict, average='macro')
      sen = str(coh) + ',' + str(acc) + ',' + str(pre) + ',' + str(rec) + ',' + str(f1)
      #pen = open('total_2_0.5.csv', 'a')
      #pen.write('SVM,' + sub + ',' + str(i) + ',' + str(j) + ',' + sen + '\n')
      #pen.close()
      y_val = j % temporal_size
      x_val = int(j / temporal_size)
      res_val[x_val, y_val] += coh
  res_val /= 5
  plt.rcParams["font.family"] = "Times New Roman"
  ax = sns.heatmap(res_val, cmap="BuGn", vmin=0.1, vmax=0.85, square=True, annot=True)
  plt.savefig('fig/4.5_0.5/' + sub + '.png', format='png', dpi=1000)
  plt.close()
  print('abc')



      
    
    

def load_fbcsp(path='data/A01T.npz'):
  data = scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/fbcsp_rev/A01T.mat')
  x, y = Competition.load_one_data(path)
  x_ = data['csp_tot']
  
  mi_idx = MIBIF.fb_mibif_with_csp(x, y, x_)
  optimal_csp = np.transpose(x_[:,:,mi_idx])
  y = np.array(BCI.lab_inv_translator(y, 4))
  kv = BCI.gen_kv_idx(y, 10)
  for train_idx, test_idx in kv:
    print('1')
    x_train, y_train = optimal_csp[train_idx], y[train_idx]
    x_test, y_test = optimal_csp[test_idx], y[test_idx]
    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train.argmax(axis=1))
    x_train = clf.transform(x_train)
    x_test = clf.transform(x_test)
    clf2 = GaussianNB()
    clf2.fit(x_train, y_train.argmax(axis=1))
    y_predict = clf2.predict(x_test)
    
    cohen_score = cohen_kappa_score(y_predict, y_test.argmax(axis=1))
    score = clf2.score(x_test, y_test.argmax(axis=1))
    print('pen!')
    pen = open("cohen_score_fb.csv", 'a')
    pen.write(str(cohen_score) + ',' + str(score) + '\n')
  print('abc')

def test_api(sub='1'):
  import matplotlib.pyplot as plt
  plt.rcParams["font.family"] = "Times New Roman"
  import seaborn as sns; sns.set()
  path = 'data/A0'+ sub + 'T.npz'
  data = scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/tscsp_rev/A0' + sub + 'T.mat')
  x, y_ = Competition.load_one_data(path)
  x_ = data['csp_2']
  
  #mi_idx = MIBIF.fb_mibif_with_csp(x, y, x_)]
  #import feature_selection as FS
  #mi_idx = FS.lsvm_filter(x_, y)
  res_val = np.zeros((9,5))
  for i in range(0, 45):
    mi_idx = i
    optimal_csp = np.transpose(x_[:,:,mi_idx])
    y = np.array(BCI.lab_inv_translator(y_, 4))
    kv = BCI.gen_kv_idx(y, 10)
    for train_idx, test_idx in kv:
      x_train, y_train = optimal_csp[train_idx], y[train_idx]
      x_test, y_test = optimal_csp[test_idx], y[test_idx]
      clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      clf2.fit(x_train, y_train.argmax(axis=1))
      y_predict = clf2.predict(x_test)
      cohen_score = cohen_kappa_score(y_predict, y_test.argmax(axis=1))
      x_val = i % 9
      y_val = int(i / 9)
      res_val[x_val, y_val] += cohen_score
    
  res_val = res_val / 10
  plt.rcParams["font.family"] = "Times New Roman"
  ax = sns.heatmap(res_val, cmap="BuGn", vmin=0.5, vmax=0.85, square=True, annot=True)
  plt.savefig('fig/' + sub + '_2.eps', format='eps', dpi=1000)
  plt.close()
  print('abc')

def load_tscsp(sub='1'):
  path = 'data/A0'+ sub + 'T.npz'
  data = scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/tscsp_rev/A0' + sub + 'T.mat')
  x, y = Competition.load_one_data(path)
  x_ = data['csp_2']
  
  import feature_selection as FS
  mi_idx = FS.lsvm_filter(x_, y)
  optimal_csp = np.transpose(x_[:,:,mi_idx])
  y = np.array(BCI.lab_inv_translator(y, 4))
  kv = BCI.gen_kv_idx(y, 10)
  for train_idx, test_idx in kv:
    x_train, y_train = optimal_csp[train_idx], y[train_idx]
    x_test, y_test = optimal_csp[test_idx], y[test_idx]
    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train.argmax(axis=1))
    x_train = clf.transform(x_train)
    x_test = clf.transform(x_test)
    clf2 = GaussianNB()
    clf2.fit(x_train, y_train.argmax(axis=1))
    y_predict = clf2.predict(x_test)
    
    cohen_score = cohen_kappa_score(y_predict, y_test.argmax(axis=1))
    score = clf2.score(x_test, y_test.argmax(axis=1))
    pen = open("FE_LSVM_sidi0.csv", 'a')
    pen.write(sub + ',' + str(cohen_score) + ',' + str(score) + ',' + str(mi_idx) + '\n')
  print('abc')

def load_tscsp_nested(sub='1'):
  path = 'data/A0'+ sub + 'T.npz'
  data = scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/tscsp_rev/A0' + sub + 'T.mat')
  x_, y = Competition.load_one_data(path)
  x = data['csp_2']
  import feature_selection as FS
  y = np.array(BCI.lab_inv_translator(y, 4))
  kv = BCI.gen_kvt_idx(y, 10)
  for train_idx, valid_idx, test_idx in kv:
    x_train, y_train = x[:,train_idx,:], y[train_idx]
    x_valid, y_valid = x[:,valid_idx,:], y[valid_idx]
    x_test, y_test = x[:,test_idx,:], y[test_idx]
    opt_idx = FS.lsvm_filter_pp(x_train, y_train, x_valid, y_valid)

    x_train = np.transpose(x_train[:,:,opt_idx])
    x_test = np.transpose(x_test[:,:,opt_idx])

    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train.argmax(axis=1))
    x_train = clf.transform(x_train)
    x_test = clf.transform(x_test)
    clf2 = GaussianNB()
    clf2.fit(x_train, y_train.argmax(axis=1))
    y_predict = clf2.predict(x_test)
    
    cohen_score = cohen_kappa_score(y_predict, y_test.argmax(axis=1))
    score = clf2.score(x_test, y_test.argmax(axis=1))
    pen = open("FE_LSVM_nested_rev2.csv", 'a')
    pen.write(sub + ',' + str(cohen_score) + ',' + str(score) + ',' + str(opt_idx) + '\n')
  print('abc')


def load_tscsp_nested2(sub='1'):
  path = 'data/A0'+ sub + 'T.npz'
  data = scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/tscsp_rev/A0' + sub + 'T.mat')
  x_, y = Competition.load_one_data(path)
  x = data['csp_2']
  import feature_selection as FS
  y = np.array(BCI.lab_inv_translator(y, 4))
  kv = BCI.gen_kv_idx(y, 10)
  accs = []; pres = []; recs = []; f1s = []; cohens = [];
  conf1 = np.zeros((4, 4))
  conf2 = np.zeros((4, 4))
  for train_idx, test_idx in kv:
    x_train, y_train = x[:,train_idx,:], y[train_idx]
    x_test, y_test = x[:,test_idx,:], y[test_idx]
    opt_idx = FS.lsvm_filter_pp2(x_train, y_train)

    x_train = np.transpose(x_train[:,:,opt_idx])
    x_test = np.transpose(x_test[:,:,opt_idx])

    #clf = LinearDiscriminantAnalysis()
    #clf.fit(x_train, y_train.argmax(axis=1))
    #x_train = clf.transform(x_train)
    #x_test = clf.transform(x_test)
    #clf2 = GaussianNB()
    #clf2.fit(x_train, y_train.argmax(axis=1))
    #y_predict = clf2.predict(x_test)
    clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    from sklearn.svm import SVC
    clf2 = SVC(kernel='rbf')

    #clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf1.fit(x_train, y_train.argmax(axis=1))
    clf2.fit(x_train, y_train.argmax(axis=1))
    y_predict1 = clf1.predict(x_test)
    y_predict2 = clf2.predict(x_test)

    from sklearn.metrics import confusion_matrix
    confusion1 = confusion_matrix(y_test.argmax(axis=1), y_predict1)
    confusion2 = confusion_matrix(y_test.argmax(axis=1), y_predict2)

    conf1 += confusion1
    conf2 += confusion2



def make_topos(sub='1'):
  optimal_bank = [20,30,19,9,9,20,20,19,22]
  import matplotlib.pyplot as plt
  plt.rcParams["font.family"] = "Times New Roman"
  path = 'data/A0'+ sub + 'T.npz'
  data = scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/tscsp_rev/A0' + sub + 'T.mat')
  x, y_ = Competition.load_one_data(path)
  x_ = data['csp_2']
  import feature_selection as FS
  y = np.array(BCI.lab_inv_translator(y, 4))
  kv = BCI.gen_kv_idx(y, 10)
  x[:,:,optimal_bank[int(sub) - 1]]
  for train_idx, test_idx in kv:
    x_train, y_train = x[:,train_idx,:], y[train_idx]
    x_test, y_test = x[:,test_idx,:], y[test_idx]
    opt_idx = FS.lsvm_filter_pp2(x_train, y_train)

    x_train = np.transpose(x_train[:,:,opt_idx])
    x_test = np.transpose(x_test[:,:,opt_idx])
      
    
  res_val = res_val / 10
  plt.rcParams["font.family"] = "Times New Roman"
  ax = sns.heatmap(res_val, cmap="BuGn", vmin=0.5, vmax=0.85, square=True, annot=True)
  plt.savefig('fig/' + sub + '_2.eps', format='eps', dpi=1000)
  plt.close()
  print('abc')

def RCNN_batch(sub = '1', epoch = 100):
  path = 'data/A0'+ sub + 'T.npz'
  data = scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/tscsp_rev/A0' + sub + 'T.mat')
  x_, y = Competition.load_one_data(path)
  x = data['csp_2']
  import feature_selection as FS
  y = np.array(BCI.lab_inv_translator(y, 4))
  kv = BCI.gen_kv_idx(y, 10)
  import RCNN, data_trans
  fold = 1
  for train_idx, test_idx in kv:
    x_train, y_train = x[:,train_idx,:], y[train_idx]
    x_test, y_test = x[:,test_idx,:], y[test_idx]
    x_train = data_trans.x_translator(x_train)
    x_test = data_trans.x_translator(x_test)
    model = RCNN.create_model2((48, 45, 1))
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, batch_size = 10)
    metrics = model.evaluate(x_test, y_test)
    pen = open('rcnn_competition_2.csv', 'a')
    pen.write('RCNN,' + sub + ',' + str(fold) + ',' + str(epoch) + ',' + str(metrics[1]) + '\n')
    pen.close()
    fold += 1


    

if __name__ == '__main__':
  #new_numpy_to_mat()
  #load_new_mat()
  #gen_fbcsp()
  #load_fbcsp()
  #gen_tscsp()

  for i in range(1, 10):
    #gen_tscsp(str(i))
    #load_tscsp(str(i))
    #classification(str(i))
    #gen_tscsp(str(i))
    gen_tscsp_trick(str(i))
