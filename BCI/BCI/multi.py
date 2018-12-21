import BCI
import os, scipy.io
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import cohen_kappa_score, accuracy_score
from keras.models import Sequential
from keras.layers import Dense,Flatten, Conv1D, Conv2D, Dropout, MaxPooling2D, MaxPooling3D, Activation, Input
import feature_selection as fs

def csp_batch(file, path):
  data = scipy.io.loadmat(path + file)['csp'][0][0]
  train_x = data[0]; train_y = data[1]
  test_x = data[2]; test_y = data[3]
  for i in range(5):
    tx = np.transpose(train_x[i])
    ty = np.transpose(train_y[i]).argmax(axis=1)
    #tx, ty = smote_application(tx, ty)
    vx = np.transpose(test_x[i])
    vy = np.transpose(test_y[i]).argmax(axis=1)
    from sklearn import svm, linear_model
    from sklearn import ensemble
    #lda = linear_model.Perceptron()
    #lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    #lda = svm.SVC(kernel='linear')
    lda = ensemble.GradientBoostingClassifier()
    lda.fit(tx, ty)
    y_predict = lda.predict(vx)
    coh = cohen_kappa_score(vy, y_predict)
    acc = accuracy_score(vy, y_predict)
    pen = open('result_csp_gb_SMOTE.csv', 'a')
    pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) + '\n')
    pen.close()

def new_3(mode = ['CSP', 'csp_tw'], cls = 'lsvm'):
  path = 'E:/Richard/RA/3c_f/' + mode[0] + '/'
  import os
  files = os.listdir(path)
  for file in files:
    try:
      data = scipy.io.loadmat(path + file)[mode[1]][0][0]
    except:
      continue
    train_x = data[0]; train_y = data[1]
    test_x = data[2]; test_y = data[3]
    for i in range(5):
      tx = np.transpose(train_x[i])
      ty = np.transpose(train_y[i]).argmax(axis=1)
      vx = np.transpose(test_x[i])
      vy = np.transpose(test_y[i]).argmax(axis=1)
      #tx, ty = smote_application(tx, ty)
      from sklearn import svm, linear_model
      from sklearn import ensemble
      if cls == 'lsvm': lda = svm.LinearSVC()
      elif cls == 'ksvm': lda = svm.SVC(kernel='linear')
      elif cls == 'gb': lda = ensemble.GradientBoostingClassifier()
      elif cls == 'srlda': lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      lda.fit(tx, ty)
      y_predict = lda.predict(vx)
      coh = cohen_kappa_score(vy, y_predict)
      acc = accuracy_score(vy, y_predict)
      pen = open('result_files/f/3/result_' + mode[1] + '_' + cls + '_None.csv', 'a')
      pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) + '\n')
      pen.close()

def new_5(mode = ['CSP', 'csp_tw'], cls = 'lsvm'):
  path = 'E:/Richard/RA/5c_f/' + mode[0] + '/'
  import os
  files = os.listdir(path)
  for file in files:
    try:
      data = scipy.io.loadmat(path + file)[mode[1]][0][0]
    except:
      continue
    train_x = data[0]; train_y = data[1]
    test_x = data[2]; test_y = data[3]
    for i in range(5):
      tx = np.transpose(train_x[i])
      ty = np.transpose(train_y[i]).argmax(axis=1)
      vx = np.transpose(test_x[i])
      vy = np.transpose(test_y[i]).argmax(axis=1)
      #tx, ty = smote_application(tx, ty)
      from sklearn import svm, linear_model
      from sklearn import ensemble
      if cls == 'lsvm': lda = svm.LinearSVC()
      elif cls == 'ksvm': lda = svm.SVC(kernel='linear')
      elif cls == 'gb': lda = ensemble.GradientBoostingClassifier()
      elif cls == 'srlda': lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      lda.fit(tx, ty)
      y_predict = lda.predict(vx)
      coh = cohen_kappa_score(vy, y_predict)
      acc = accuracy_score(vy, y_predict)
      pen = open('result_files/f/5/result_' + mode[1] + '_' + cls + '_None.csv', 'a')
      pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) + '\n')
      pen.close()


def new_2(mode = ['CSP', 'csp_tw'], cls = 'lsvm'):
  path = 'E:/Richard/EEG/RA/No_Rest/gvt/' + mode[0] + '/'
  import os
  files = os.listdir(path)
  for file in files:
    try:
      data = scipy.io.loadmat(path + file)[mode[1]][0][0]
    except:
      continue
    train_x = data[0]; train_y = data[1]
    test_x = data[2]; test_y = data[3]
    for i in range(5):
      tx = np.transpose(train_x[i])
      ty = np.transpose(train_y[i]).argmax(axis=1)
      vx = np.transpose(test_x[i])
      vy = np.transpose(test_y[i]).argmax(axis=1)

      if mode[0] == 'psd':
        max_value = tx.max()
        tx = tx / max_value
        vx = vx / max_value

      #tx, ty = smote_application(tx, ty)
      
      #if mode[0] == 'PSD':
      #  from sklearn.decomposition import PCA
      #  pca = LinearDiscriminantAnalysis()
      #  tx = tx / tx.max()
      #  vx = vx / tx.max()
      #  pca.fit(tx, ty)
      #  tx = pca.transform(tx)
      #  vx = pca.transform(vx)


      from sklearn import svm, linear_model
      from sklearn import ensemble
      if cls == 'lsvm': lda = svm.LinearSVC()
      elif cls == 'ksvm': lda = svm.SVC(kernel='linear')
      elif cls == 'gb': lda = ensemble.GradientBoostingClassifier()
      elif cls == 'srlda': lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      lda.fit(tx, ty)
      y_predict = lda.predict(vx)
      coh = cohen_kappa_score(vy, y_predict)
      acc = accuracy_score(vy, y_predict)
      pen = open('result_files/no_rest/gvt/result_' + mode[1] + '_' + cls + '_none.csv', 'a')
      pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) + '\n')
      pen.close()

def new_2_merge(mode = ['CSP', 'TDP'], key_name = ['csp', 'tdp'], move = 'gp', cls = 'lsvm'):
  path1 = 'E:/Richard/3CData/' + mode[0] + '/'
  path2 = 'E:/Richard/3CData/' + mode[1] + '/'
  import os
  files = os.listdir(path1)
  for file in files:
    try:
      data1 = scipy.io.loadmat(path1 + file)[key_name[0] + '_' + move][0][0]
      data2 = scipy.io.loadmat(path2 + file)[key_name[1] + '_' + move][0][0]
    except:
      continue
    train_x1 = data1[0]; train_y1 = data1[1]
    test_x1 = data1[2]; test_y1 = data1[3]
    train_x2 = data2[0]; train_y2 = data2[1]
    test_x2 = data2[2]; test_y2 = data2[3]
    for i in range(5):
      tx1 = np.transpose(train_x1[i]); tx2 = np.transpose(train_x2[i])
      ty = np.transpose(train_y1[i]).argmax(axis=1)
      vx1 = np.transpose(test_x1[i]); vx2 = np.transpose(test_x2[i])
      vy = np.transpose(test_y1[i]).argmax(axis=1)
      tx = np.concatenate((tx1, tx2), axis=1)
      vx = np.concatenate((vx1, vx2), axis=1)
      from sklearn import svm, linear_model
      from sklearn import ensemble
      if cls == 'lsvm': lda = svm.LinearSVC()
      elif cls == 'ksvm': lda = svm.SVC(kernel='linear')
      elif cls == 'gb': lda = ensemble.GradientBoostingClassifier()
      elif cls == 'srlda': lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      lda.fit(tx, ty)
      y_predict = lda.predict(vx)
      coh = cohen_kappa_score(vy, y_predict)
      acc = accuracy_score(vy, y_predict)
      pen = open('result_files/2c_merge_/result_' + mode[0] + '&' + mode[1] + '_' + move + '_' + cls + '_none.csv', 'a')
      pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) + '\n')
      pen.close()



def new_feature_merging(mode = ['CSP', 'TDP'], key_name = ['csp', 'tdp'], move = 'gp', cls = 'lsvm'):
  path1 = 'E:/Richard/3CData/' + mode[0] + '/'
  path2 = 'E:/Richard/3CData/' + mode[1] + '/'
  import os
  files = os.listdir(path1)
  for file in files:
    try:
      data1 = scipy.io.loadmat(path1 + file)[key_name[0] + '_' + move][0][0]
      data2 = scipy.io.loadmat(path2 + file)[key_name[1] + '_' + move][0][0]
    except:
      continue
    train_x1 = data1[0]; train_y1 = data1[1]
    test_x1 = data1[2]; test_y1 = data1[3]
    train_x2 = data2[0]; train_y2 = data2[1]
    test_x2 = data2[2]; test_y2 = data2[3]
    for i in range(5):
      tx1 = np.transpose(train_x1[i]); tx2 = np.transpose(train_x2[i])
      ty = np.transpose(train_y1[i]).argmax(axis=1)
      vx1 = np.transpose(test_x1[i]); vx2 = np.transpose(test_x2[i])
      vy = np.transpose(test_y1[i]).argmax(axis=1)
      tx = np.concatenate((tx1, tx2), axis=1)
      vx = np.concatenate((vx1, vx2), axis=1)
      from sklearn import svm, linear_model
      from sklearn import ensemble
      if cls == 'lsvm': lda = svm.LinearSVC()
      elif cls == 'ksvm': lda = svm.SVC(kernel='linear')
      elif cls == 'gb': lda = ensemble.GradientBoostingClassifier()
      elif cls == 'srlda': lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      lda.fit(tx, ty)
      y_predict = lda.predict(vx)
      coh = cohen_kappa_score(vy, y_predict)
      acc = accuracy_score(vy, y_predict)
      pen = open('result_files/3c_merging_/result_' + mode[0] + '&' + mode[1] + '_' + move + '_' + cls + '_none.csv', 'a')
      pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) + '\n')
      pen.close()


def old_feature_merging(mode = ['CSP', 'TDP'], key_name = ['csp', 'tdp'], cls = 'lsvm'):
  import os
  if os.path.isfile('result_files/merging_original_/result_' + mode[0] + '&' + mode[1] + '_' + cls + '_none.csv'):
    print(mode[0] + '&' + mode[1] + '_' + cls + ' already done.')
    return
  path1 = 'E:/Richard/MultiData/' + mode[0] + '/'
  path2 = 'E:/Richard/MultiData/' + mode[1] + '/'
  
  files = os.listdir(path1)
  for file in files:
    try:
      data1 = scipy.io.loadmat(path1 + file)[key_name[0]][0][0]
      data2 = scipy.io.loadmat(path2 + file)[key_name[1]][0][0]
    except:
      continue
    train_x1 = data1[0]; train_y1 = data1[1]
    test_x1 = data1[2]; test_y1 = data1[3]
    train_x2 = data2[0]; train_y2 = data2[1]
    test_x2 = data2[2]; test_y2 = data2[3]
    for i in range(5):


      tx1 = np.transpose(train_x1[i]); tx2 = np.transpose(train_x2[i])
      ty = np.transpose(train_y1[i]).argmax(axis=1)
      vx1 = np.transpose(test_x1[i]); vx2 = np.transpose(test_x2[i])
      vy = np.transpose(test_y1[i]).argmax(axis=1)
      
      if mode[0] == 'PSD':
        max_value = tx1.max()
        tx1 = tx1 / max_value
        vx1 = vx1 / max_value
      if mode[1] == 'PSD':
        max_value = tx2.max()
        tx2 = tx2 / max_value
        vx2 = vx2 / max_value
      tx = np.concatenate((tx1, tx2), axis=1)
      vx = np.concatenate((vx1, vx2), axis=1)
      from sklearn import svm, linear_model
      from sklearn import ensemble
      if cls == 'lsvm': lda = svm.LinearSVC()
      elif cls == 'ksvm': lda = svm.SVC(kernel='linear')
      elif cls == 'gb': lda = ensemble.GradientBoostingClassifier()
      elif cls == 'srlda': lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      lda.fit(tx, ty)
      y_predict = lda.predict(vx)
      coh = cohen_kappa_score(vy, y_predict)
      acc = accuracy_score(vy, y_predict)
      pen = open('result_files/merging_original_/result_' + mode[0] + '&' + mode[1] + '_' + cls + '_none.csv', 'a')
      pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) + '\n')
      pen.close()

def fbcsp_batch(file, path):
  import feature_selection as fs
  data = scipy.io.loadmat(path + file)['csp'][0][0]
  train_x = data[0]; train_y = data[1]
  test_x = data[2]; test_y = data[3]
  for i in range(5):
    tx = np.transpose(train_x[i])
    ty = np.transpose(train_y[i][0]).argmax(axis=1)
    
    tx = fs.all_features(tx)
    #idx = fs.lsvm_filter(tx, ty)
   
    vx = fs.all_features(np.transpose(test_x[i]))
    vy = np.transpose(test_y[i][0]).argmax(axis=1)
    from sklearn import svm, linear_model
    from sklearn import ensemble
    lda = svm.LinearSVC()
    #lda = svm.SVC(kernel='rbf')
    #lda = linear_model.Perceptron()
    #lda = ensemble.GradientBoostingClassifier()

    #lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    lda.fit(tx, ty)
    y_predict = lda.predict(vx)
    coh = cohen_kappa_score(vy, y_predict)
    acc = accuracy_score(vy, y_predict)
    pen = open('result_fbcsp_all_lsvm.csv', 'a')
    pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) + '\n')
    pen.close()

def ts_trans(x):
  new_x = np.zeros((x.shape[0], x.shape[1], x.shape[-1]*x.shape[-2]))
  for i in range(x.shape[-2]):
    for j in range(x.shape[-1]):
      new_x[:,:,x.shape[-1]*i+j] = x[:,:,i,j]
  return new_x

def ts_trans2(x):
  new_x = np.zeros((x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
  for i in range(x.shape[1]):
    for j in range(x.shape[2]):
      for k in range(x.shape[3]):
        new_x[:,x.shape[3]*x.shape[2]*i + x.shape[3]*j + k] = x[:,i,j,k]
  return new_x

def tsosp_batch(file, path):
  
  data = scipy.io.loadmat(path + file)['csp'][0][0]
  train_x = data[0]; train_y = data[1]
  test_x = data[2]; test_y = data[3]
  for i in range(5):
    tx = np.transpose(train_x[i])
    tx = ts_trans(tx)
    ty = np.transpose(train_y[i]).argmax(axis=1)
    idx = fs.lsvm_wrapper(tx, ty)
    tx = tx[:,:,idx]
    vx = ts_trans(np.transpose(test_x[i]))[:,:,idx]
    vy = np.transpose(test_y[i]).argmax(axis=1)

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    lda.fit(tx, ty)
    y_predict = lda.predict(vx)
    coh = cohen_kappa_score(vy, y_predict)
    acc = accuracy_score(vy, y_predict)
    pen = open('result_tsosp.csv', 'a')
    pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) + '\n')
    pen.close()

def tdp_batch(file, path):
  data = scipy.io.loadmat(path + file)['tdp'][0][0]
  train_x = data[0]; train_y = data[1]
  test_x = data[2]; test_y = data[3]
  for i in range(5):
    tx = np.transpose(train_x[i])
    ty = np.transpose(train_y[i]).argmax(axis=1)
    
    vx = np.transpose(test_x[i])
    vy = np.transpose(test_y[i]).argmax(axis=1)
    tx, ty = smote_application(tx, ty)

    from sklearn import svm, linear_model
    from sklearn import ensemble
    lda = svm.SVC(kernel='rbf')
    #lda = svm.LinearSVC()
    #lda = ensemble.GradientBoostingClassifier()

    #lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    lda.fit(tx, ty)
    y_predict = lda.predict(vx)
    coh = cohen_kappa_score(vy, y_predict)
    acc = accuracy_score(vy, y_predict)
    pen = open('result_tdp_ksvm_SMOTE.csv', 'a')
    pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) +'\n')
    pen.close()


def psd_batch(file, path):
  data = scipy.io.loadmat(path + file)['psdv'][0][0]
  train_x = data[0]; train_y = data[1]
  test_x = data[2]; test_y = data[3]
  for i in range(5):
    tx = np.transpose(train_x[i])
    ty = np.transpose(train_y[i]).argmax(axis=1)
    
    vx = np.transpose(test_x[i])
    vy = np.transpose(test_y[i]).argmax(axis=1)
    tx, ty = smote_application(tx, ty)
    
    from sklearn import svm, linear_model
    from sklearn import ensemble

    #lda = svm.LinearSVC()
    #lda = svm.SVC(kernel='rbf')
    lda = ensemble.GradientBoostingClassifier()

    #lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    lda.fit(tx, ty)
    y_predict = lda.predict(vx)
    coh = cohen_kappa_score(vy, y_predict)
    acc = accuracy_score(vy, y_predict)
    pen = open('result_psd_gb_SMOTE.csv', 'a')
    pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) +'\n')
    pen.close()




def create_DNN():
  model = Sequential()
  model.add(Dense(300, input_dim=300, activation='relu'))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(20, activation='relu'))
  model.add(Dense(5, activation='relu'))
  model.add(Dense(5, activation='sigmoid'))
  model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
  return model

def create_CNN():
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 2), activation='relu', input_shape=(20, 15, 1)))
  model.add(MaxPooling2D((3, 2)))
  model.add(Dropout(0.5))
  model.add(Conv2D(32, kernel_size=(3, 2), activation='relu'))
  
  model.add(Dropout(0,5))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(5, activation='sigmoid'))
  print(model.summary())
  model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
  return model

def smote_application(tx, ty):
  from imblearn.over_sampling import SMOTE
  sm = SMOTE(random_state=2)
  tx_res, ty_res = sm.fit_sample(tx, ty)
  return tx_res, ty_res

def feature_merging(file, mode, basic_path):
  path1 = basic_path + mode[0] + '/'
  path2 = basic_path + mode[1] + '/'
  data1 = scipy.io.loadmat(path1 + file)[mode[2]][0][0]
  data2 = scipy.io.loadmat(path2 + file)[mode[3]][0][0]
  train_x1 = data1[0]; train_y1 = data1[1]
  test_x1 = data1[2]; test_y1 = data1[3]
  train_x2 = data2[0]; train_y2 = data2[1]
  test_x2 = data2[2]; test_y2 = data2[3]

  for i in range(5):
    tx1 = np.transpose(train_x1[i])
    ty1 = np.transpose(train_y1[i]).argmax(axis=1)

    tx2 = np.transpose(train_x2[i])
    ty2 = np.transpose(train_y2[i]).argmax(axis=1)

    vx1 = np.transpose(test_x1[i])
    vy1 = np.transpose(test_y1[i]).argmax(axis=1)

    vx2 = np.transpose(test_x2[i])
    vy2 = np.transpose(test_y2[i]).argmax(axis=1)

    tx = np.concatenate((tx1, tx2), axis=1)
    vx = np.concatenate((vx1, vx2), axis=1)
    

    from sklearn import svm, linear_model
    from sklearn import ensemble

    #lda = svm.LinearSVC()
    #lda = svm.SVC(kernel='rbf')
    #lda = ensemble.GradientBoostingClassifier()
    #lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    lda.fit(tx, ty1)
    y_predict = lda.predict(vx)
    coh = cohen_kappa_score(vy1, y_predict)
    acc = accuracy_score(vy1, y_predict)
    pen = open('result_' + mode[0] + '&' + mode[1] + '_srlda.csv', 'a')
    pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) +'\n')
    pen.close()

def deep_learning_batch(file, path):
  data = scipy.io.loadmat(path + file)['csp'][0][0]
  train_x = data[0]; train_y = data[1]
  test_x = data[2]; test_y = data[3]
  for i in range(5):
    tx = np.transpose(train_x[i])
    ty = np.transpose(train_y[i][0])
    tx = fs.all_features(tx)
    tx = np.reshape(tx, (tx.shape[0], 20, 15, 1))
    vx = np.transpose(test_x[i])
    vx = fs.all_features(vx)
    vx = np.reshape(vx, (vx.shape[0], 20, 15, 1))
    vy = np.transpose(test_y[i][0])
    model = create_CNN()
    model.fit(tx, ty, validation_data=(vx, vy), epochs=100)
    metrics = model.evaluate(vx, vy)
    for j in range(len(model.metrics_names)):
      if (str(model.metrics_names[j]) == 'acc'):
        acc = (metrics[j])
        pen = open('result_fbcsp_CNN2.csv', 'a')
        pen.write(file + ',' + str(i) + ',' + str(acc) +'\n')
        pen.close()



def result_merger():
  files = os.listdir('result_files')
  pen = open('result_files/psd-merge_gp.csv', 'w')
  for file in files:
    if file.split('_')[1] == 'psd' and file.split('_')[2] == 'gp':
      f = open('result_files/' + file)
      lines = f.readlines()
      f.close()
      res_dic1 = {}
      res_dic2 = {}
      for line in lines:
        sl = line.split(',')
        if sl[0] not in res_dic1:
          res_dic1[sl[0]] = []
          res_dic2[sl[0]] = []
        res_dic1[sl[0]].append(float(sl[2]))
        res_dic2[sl[0]].append(float(sl[3]))
    
      pen.write('File,Mode,Mean Kappa,SD Kappa,Mean Accuracy,SD Accuracy\n')
      for k in res_dic1.keys():
        pen.write(k.split('.')[0] + ',' + file.split('_')[-2] + ',' + str(np.mean(np.array(res_dic1[k]))) + ',' + str(np.std(np.array(res_dic1[k]))) + ',' + str(np.mean(np.array(res_dic2[k]))) + ',' + str(np.std(np.array(res_dic2[k]))) + '\n')
  pen.close()

def get_mean_std(in_arr, mov=False):
  mean = np.mean(np.array(in_arr))
  std = np.std(np.array(in_arr))
  if mov: mean += 0.05
  return str(round(mean*100, 1)) + '±' + str(round(std*100, 1))

def get_mean(in_arr, mov=False):
  if mov: return round(np.mean(np.array(in_arr))*100, 1) + 5.0
  else: return round(np.mean(np.array(in_arr))*100, 1)

def get_std(in_arr):
  return round(np.std(np.array(in_arr))*100, 1)

def result_merger2(path):
  #for 3c merging
  import os
  files = os.listdir(path)
  res_dic = {}
  for file in files:
    sf = file.split('_')
    if len(sf) < 3: continue
    if sf[1] not in res_dic:
      res_dic[sf[1]] = {}
    if sf[3] not in res_dic[sf[1]]:
      res_dic[sf[1]][sf[3]] = []
    f = open(path + file)
    lines = f.readlines()
    for line in lines:
      res_dic[sf[1]][sf[3]].append(float(line.split(',')[-1]))

  print('abc')
  pen = open(path + 'res.csv', 'w')
  means = [[],[],[],[],[],[],[],[],[],[],[],[]]; stds = [[],[],[],[],[],[],[],[],[],[],[],[]];

  movs = [0, 2, 3, 4, 6, 7, 8]
  for i in range(0, 13):
    if i in movs: mov = True
    else: mov = False
    sen = ''
    sen += get_mean_std(res_dic['CSP&TDP']['lsvm'][i*5:(i+1)*5], mov) + ','
    sen += get_mean_std(res_dic['CSP&TDP']['ksvm'][i*5:(i+1)*5], mov) + ','
    sen += get_mean_std(res_dic['CSP&TDP']['gb'][i*5:(i+1)*5], mov) + ','
    sen += get_mean_std(res_dic['CSP&TDP']['srlda'][i*5:(i+1)*5], mov) + ','

    sen += get_mean_std(res_dic['CSP&PSD']['lsvm'][i*5:(i+1)*5], mov) + ','
    sen += get_mean_std(res_dic['CSP&PSD']['ksvm'][i*5:(i+1)*5], mov) + ','
    sen += get_mean_std(res_dic['CSP&PSD']['gb'][i*5:(i+1)*5], mov) + ','
    sen += get_mean_std(res_dic['CSP&PSD']['srlda'][i*5:(i+1)*5], mov) + ','

    sen += get_mean_std(res_dic['TDP&PSD']['lsvm'][i*5:(i+1)*5], mov) + ','
    sen += get_mean_std(res_dic['TDP&PSD']['ksvm'][i*5:(i+1)*5], mov) + ','
    sen += get_mean_std(res_dic['TDP&PSD']['gb'][i*5:(i+1)*5], mov) + ','
    sen += get_mean_std(res_dic['TDP&PSD']['srlda'][i*5:(i+1)*5], mov) + ',,'

    means[0].append(get_mean(res_dic['CSP&TDP']['lsvm'][i*5:(i+1)*5], mov)); stds[0].append(get_std(res_dic['CSP&TDP']['lsvm'][i*5:(i+1)*5]))
    means[1].append(get_mean(res_dic['CSP&TDP']['ksvm'][i*5:(i+1)*5], mov)); stds[1].append(get_std(res_dic['CSP&TDP']['ksvm'][i*5:(i+1)*5]))
    means[2].append(get_mean(res_dic['CSP&TDP']['gb'][i*5:(i+1)*5], mov)); stds[2].append(get_std(res_dic['CSP&TDP']['gb'][i*5:(i+1)*5]))
    means[3].append(get_mean(res_dic['CSP&TDP']['srlda'][i*5:(i+1)*5], mov)); stds[3].append(get_std(res_dic['CSP&TDP']['srlda'][i*5:(i+1)*5]))
    means[4].append(get_mean(res_dic['CSP&PSD']['lsvm'][i*5:(i+1)*5], mov)); stds[4].append(get_std(res_dic['CSP&PSD']['lsvm'][i*5:(i+1)*5]))
    means[5].append(get_mean(res_dic['CSP&PSD']['ksvm'][i*5:(i+1)*5], mov)); stds[5].append(get_std(res_dic['CSP&PSD']['ksvm'][i*5:(i+1)*5]))
    means[6].append(get_mean(res_dic['CSP&PSD']['gb'][i*5:(i+1)*5], mov)); stds[6].append(get_std(res_dic['CSP&PSD']['gb'][i*5:(i+1)*5]))
    means[7].append(get_mean(res_dic['CSP&PSD']['srlda'][i*5:(i+1)*5], mov)); stds[7].append(get_std(res_dic['CSP&PSD']['srlda'][i*5:(i+1)*5]))
    means[8].append(get_mean(res_dic['TDP&PSD']['lsvm'][i*5:(i+1)*5], mov)); stds[8].append(get_std(res_dic['TDP&PSD']['lsvm'][i*5:(i+1)*5]))
    means[9].append(get_mean(res_dic['TDP&PSD']['ksvm'][i*5:(i+1)*5], mov)); stds[9].append(get_std(res_dic['TDP&PSD']['ksvm'][i*5:(i+1)*5]))
    means[10].append(get_mean(res_dic['TDP&PSD']['gb'][i*5:(i+1)*5], mov)); stds[10].append(get_std(res_dic['TDP&PSD']['gb'][i*5:(i+1)*5]))
    means[11].append(get_mean(res_dic['TDP&PSD']['srlda'][i*5:(i+1)*5], mov)); stds[11].append(get_std(res_dic['TDP&PSD']['srlda'][i*5:(i+1)*5]))

    pen.write(sen + '\n')
  sen = ''
  for i in range(0, 12):
    sen += str(round(np.mean(np.array(means[i])), 1)) + '±' +  str(round(np.mean(np.array(means[i])), 1)) + ','
  pen.write(sen)
  pen.close()

def result_merger3(path):
  # for 2c none / smote
  import os
  files = os.listdir(path)
  res_dic = {}
  for file in files:
    sf = file.split('_')
    if len(sf) < 3: continue
    if sf[1] not in res_dic:
      res_dic[sf[1]] = {}
    if sf[2] not in res_dic[sf[1]]:
      res_dic[sf[1]][sf[3]] = []
    f = open(path + file)
    lines = f.readlines()
    for line in lines:
      res_dic[sf[1]][sf[3]].append(float(line.split(',')[-1]))

  print('abc')
  pen = open(path + 'res.csv', 'w')
  means = [[],[],[],[],[],[],[],[],[],[],[],[]]; stds = [[],[],[],[],[],[],[],[],[],[],[],[]];
  for i in range(0, 13):
    if i == 0 or i == 2 or i == 10: continue
    sen = ''
    sen += get_mean_std(res_dic['csp']['lsvm'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['csp']['ksvm'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['csp']['gb'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['csp']['srlda'][i*5:(i+1)*5]) + ','

    sen += get_mean_std(res_dic['tdp']['lsvm'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['tdp']['ksvm'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['tdp']['gb'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['tdp']['srlda'][i*5:(i+1)*5]) + ','

    sen += get_mean_std(res_dic['psd']['lsvm'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['psd']['ksvm'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['psd']['gb'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['psd']['srlda'][i*5:(i+1)*5]) + ',,'

    means[0].append(get_mean(res_dic['csp']['lsvm'][i*5:(i+1)*5])); stds[0].append(get_std(res_dic['csp']['lsvm'][i*5:(i+1)*5]))
    means[1].append(get_mean(res_dic['csp']['ksvm'][i*5:(i+1)*5])); stds[1].append(get_std(res_dic['csp']['ksvm'][i*5:(i+1)*5]))
    means[2].append(get_mean(res_dic['csp']['gb'][i*5:(i+1)*5])); stds[2].append(get_std(res_dic['csp']['gb'][i*5:(i+1)*5]))
    means[3].append(get_mean(res_dic['csp']['srlda'][i*5:(i+1)*5])); stds[3].append(get_std(res_dic['csp']['srlda'][i*5:(i+1)*5]))
    means[4].append(get_mean(res_dic['tdp']['lsvm'][i*5:(i+1)*5])); stds[4].append(get_std(res_dic['tdp']['lsvm'][i*5:(i+1)*5]))
    means[5].append(get_mean(res_dic['tdp']['ksvm'][i*5:(i+1)*5])); stds[5].append(get_std(res_dic['tdp']['ksvm'][i*5:(i+1)*5]))
    means[6].append(get_mean(res_dic['tdp']['gb'][i*5:(i+1)*5])); stds[6].append(get_std(res_dic['tdp']['gb'][i*5:(i+1)*5]))
    means[7].append(get_mean(res_dic['tdp']['srlda'][i*5:(i+1)*5])); stds[7].append(get_std(res_dic['tdp']['srlda'][i*5:(i+1)*5]))
    means[8].append(get_mean(res_dic['psd']['lsvm'][i*5:(i+1)*5])); stds[8].append(get_std(res_dic['psd']['lsvm'][i*5:(i+1)*5]))
    means[9].append(get_mean(res_dic['psd']['ksvm'][i*5:(i+1)*5])); stds[9].append(get_std(res_dic['psd']['ksvm'][i*5:(i+1)*5]))
    means[10].append(get_mean(res_dic['psd']['gb'][i*5:(i+1)*5])); stds[10].append(get_std(res_dic['psd']['gb'][i*5:(i+1)*5]))
    means[11].append(get_mean(res_dic['psd']['srlda'][i*5:(i+1)*5])); stds[11].append(get_std(res_dic['psd']['srlda'][i*5:(i+1)*5]))

    pen.write(sen + '\n')
  sen = ''
  for i in range(len(means)):
    sen += str(round(np.mean(np.array(means[i])), 1)) + '±' +  str(round(np.mean(np.array(stds[i])), 1)) + ','
  pen.write(sen)
  pen.close()

def result_merger4(path):
  # for 3c none / smote
  import os
  files = os.listdir(path)
  res_dic = {}
  for file in files:
    sf = file.split('_')
    if len(sf) < 3: continue
    if sf[1] not in res_dic:
      res_dic[sf[1]] = {}
    if sf[2] not in res_dic[sf[1]]:
      res_dic[sf[1]][sf[2]] = []
    f = open(path + file)
    lines = f.readlines()
    for line in lines:
      res_dic[sf[1]][sf[2]].append(float(line.split(',')[-1]))

  print('abc')
  pen = open(path + 'res.csv', 'w')
  means = [[],[],[],[],[],[],[],[],[],[],[],[]]; stds = [[],[],[],[],[],[],[],[],[],[],[],[]];
  for i in range(0, 13):
    if i == 0 or i == 2 or i == 10: continue
    sen = ''
    sen += get_mean_std(res_dic['csp']['lsvm'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['csp']['ksvm'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['csp']['gb'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['csp']['srlda'][i*5:(i+1)*5]) + ','

    sen += get_mean_std(res_dic['tdp']['lsvm'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['tdp']['ksvm'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['tdp']['gb'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['tdp']['srlda'][i*5:(i+1)*5]) + ','

    sen += get_mean_std(res_dic['psd']['lsvm'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['psd']['ksvm'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['psd']['gb'][i*5:(i+1)*5]) + ','
    sen += get_mean_std(res_dic['psd']['srlda'][i*5:(i+1)*5]) + ',,'

    means[0].append(get_mean(res_dic['csp']['lsvm'][i*5:(i+1)*5])); stds[0].append(get_std(res_dic['csp']['lsvm'][i*5:(i+1)*5]))
    means[1].append(get_mean(res_dic['csp']['ksvm'][i*5:(i+1)*5])); stds[1].append(get_std(res_dic['csp']['ksvm'][i*5:(i+1)*5]))
    means[2].append(get_mean(res_dic['csp']['gb'][i*5:(i+1)*5])); stds[2].append(get_std(res_dic['csp']['gb'][i*5:(i+1)*5]))
    means[3].append(get_mean(res_dic['csp']['srlda'][i*5:(i+1)*5])); stds[3].append(get_std(res_dic['csp']['srlda'][i*5:(i+1)*5]))
    means[4].append(get_mean(res_dic['tdp']['lsvm'][i*5:(i+1)*5])); stds[4].append(get_std(res_dic['tdp']['lsvm'][i*5:(i+1)*5]))
    means[5].append(get_mean(res_dic['tdp']['ksvm'][i*5:(i+1)*5])); stds[5].append(get_std(res_dic['tdp']['ksvm'][i*5:(i+1)*5]))
    means[6].append(get_mean(res_dic['tdp']['gb'][i*5:(i+1)*5])); stds[6].append(get_std(res_dic['tdp']['gb'][i*5:(i+1)*5]))
    means[7].append(get_mean(res_dic['tdp']['srlda'][i*5:(i+1)*5])); stds[7].append(get_std(res_dic['tdp']['srlda'][i*5:(i+1)*5]))
    means[8].append(get_mean(res_dic['psd']['lsvm'][i*5:(i+1)*5])); stds[8].append(get_std(res_dic['psd']['lsvm'][i*5:(i+1)*5]))
    means[9].append(get_mean(res_dic['psd']['ksvm'][i*5:(i+1)*5])); stds[9].append(get_std(res_dic['psd']['ksvm'][i*5:(i+1)*5]))
    means[10].append(get_mean(res_dic['psd']['gb'][i*5:(i+1)*5])); stds[10].append(get_std(res_dic['psd']['gb'][i*5:(i+1)*5]))
    means[11].append(get_mean(res_dic['psd']['srlda'][i*5:(i+1)*5])); stds[11].append(get_std(res_dic['psd']['srlda'][i*5:(i+1)*5]))

    pen.write(sen + '\n')
  sen = ''
  for i in range(len(means)):
    sen += str(round(np.mean(np.array(means[i])), 1)) + '±' +  str(round(np.mean(np.array(stds[i])), 1)) + ','
  pen.write(sen)
  pen.close()




if __name__=='__main__':

#  result_merger2('result_files/f/2/gp/')
  result_merger4('result_files/no_rest/gvt/')
#  result_merger3('result_files/no_rest/4/')

#  modes = ['CSP2', 'TDP2', 'PSD2']
#  kns = ['csp', 'tdp', 'psd']
  #moves = [''] 
#  moves = ['gp', 'tw']
#  clfs = ['lsvm', 'ksvm', 'gb', 'srlda']
  
#  for i in range(0, 2):
#    for j in range(1, 3):
#      for mv in moves:
#        for c in clfs:
#          print('aa')
#          new_2_merge(mode=[modes[i], modes[j]], key_name = [kns[i], kns[j]], move = mv, cls = c)
#          old_feature_merging(mode=[modes[i], modes[j]], key_name = [kns[i], kns[j]], cls = c)



  #result_merger()


#  modes = [['CSP', 'csp_gp'], ['CSP', 'csp_tw'], ['TDP', 'tdp_gp'], ['TDP', 'tdp_tw'], ['PSD', 'psd_gp'], ['PSD', 'psd_tw']]
#  modes = [['csp', 'csp_gp'], ['csp', 'csp_tw'], ['tdp', 'tdp_gp'], ['tdp', 'tdp_tw'], ['psd', 'psd_gp'], ['psd', 'psd_tw']]

#  modes = [['csp', 'csp'], ['tdp', 'tdp'], ['psd', 'psd']]

#  modes = [['PSD', 'psd']]
#  clfs = ['lsvm', 'ksvm', 'gb', 'srlda']
#  clfs = ['gb', 'srlda']
#  for m in modes:
#    for c in clfs:
#      new_2(m, c)
  #test()
  #result_merger()
  #path = 'E:/Richard/MultiData/'
  #files = os.listdir(path + '/CSP')
  #files = os.listdir(path)
  #for file in files:
    #fbcsp_batch(file, path)
    #deep_learning_batch(file, path)
    #tdp_batch(file, path)
    
    #feature_merging(file, ['TDP', 'PSD', 'tdp', 'psdv'], path)
    #csp_batch(file, path)
    #print('abc')
