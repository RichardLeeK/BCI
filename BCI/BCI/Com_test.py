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

def gen_tscsp(sub='1'):
  path = 'data/A0'+ sub + 'T.npz'
  x, y = Competition.load_one_data(path)
  x = np.array(x)
  step = int(len(x[0][0]) / 5)
  for i in range(0, 5):
    cur_x = x[:, :, step * i:step * (i + 1)]
    for j in range(1, 10):
      cur_cur_x = CSP.arr_bandpass_filter(cur_x, j*4, (j+1)*4, 250)
      f = one_fbcsp(cur_cur_x, y)
      scipy.io.savemat('mat/tscsp_ori/A0' + sub + 'T_' + str(i) + '_' + str(j) + '_1.mat', f[0])
      scipy.io.savemat('mat/tscsp_ori/A0' + sub + 'T_' + str(i) + '_' + str(j) + '_2.mat', f[1])
      scipy.io.savemat('mat/tscsp_ori/A0' + sub + 'T_' + str(i) + '_' + str(j) + '_3.mat', f[2])

def gen_stcsp(path='data/A01T.npz'):
  x, y = Competition.load_one_data(path)
  x = np.array(x)
  step = int(len(x[0][0]) / 5)
  for i in range(1, 10):
    cur_x = CSP.arr_bandpass_filter(x, i*4, (i+1)*4, 250)
    for j in range(0, 5):
      cur_cur_x = x[:, :, step * i:step * (i + 1)]
      f = one_fbcsp(cur_cur_x, y)
      scipy.io.savemat('mat/stcsp_ori/A01T_' + str(i) + '_' + str(j) + '_1.mat', f[0])
      scipy.io.savemat('mat/stcsp_ori/A01T_' + str(i) + '_' + str(j) + '_2.mat', f[1])
      scipy.io.savemat('mat/stcsp_ori/A01T_' + str(i) + '_' + str(j) + '_3.mat', f[2])






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
  
  #mi_idx = MIBIF.fb_mibif_with_csp(x, y, x_)]
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
    """
    cohens.append(cohen_kappa_score(y_predict, y_test.argmax(axis=1)))
    accs.append(accuracy_score(y_test.argmax(axis=1), y_predict))
    pres.append(precision_score(y_test.argmax(axis=1), y_predict, average='macro'))
    recs.append(recall_score(y_test.argmax(axis=1), y_predict, average='macro'))
    f1s.append(f1_score(y_test.argmax(axis=1), y_predict, average='macro'))
    """
   print('abc')


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
    load_tscsp_nested2(str(i))

