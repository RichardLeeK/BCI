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
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
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
  pen = open('result_files/csp-merge_.csv', 'w')
  for file in files:
    if file.split('_')[1] == 'csp':
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

def test():
  import matplotlib.pyplot as plt
  import numpy as np

  from sklearn.linear_model import MultiTaskLasso, Lasso

  rng = np.random.RandomState(42)

  # Generate some 2D coefficients with sine waves with random frequency and phase
  n_samples, n_features, n_tasks = 100, 30, 40
  n_relevant_features = 5
  coef = np.zeros((n_tasks, n_features))
  times = np.linspace(0, 2 * np.pi, n_tasks)
  for k in range(n_relevant_features):
      coef[:, k] = np.sin((1. + rng.randn(1)) * times + 3 * rng.randn(1))

  X = rng.randn(n_samples, n_features)
  Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)

  mtl = MultiTaskLasso(alpha=1.)
  mtl.fit(X, Y)
  abc = mtl.score()

  coef_lasso_ = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])
  coef_multi_task_lasso_ = MultiTaskLasso(alpha=1.).fit(X, Y).coef_

  # #############################################################################
  # Plot support and time series
  fig = plt.figure(figsize=(8, 5))
  plt.subplot(1, 2, 1)
  plt.spy(coef_lasso_)
  plt.xlabel('Feature')
  plt.ylabel('Time (or Task)')
  plt.text(10, 5, 'Lasso')
  plt.subplot(1, 2, 2)
  plt.spy(coef_multi_task_lasso_)
  plt.xlabel('Feature')
  plt.ylabel('Time (or Task)')
  plt.text(10, 5, 'MultiTaskLasso')
  fig.suptitle('Coefficient non-zero location')

  feature_to_plot = 0
  plt.figure()
  lw = 2
  plt.plot(coef[:, feature_to_plot], color='seagreen', linewidth=lw,
           label='Ground truth')
  plt.plot(coef_lasso_[:, feature_to_plot], color='cornflowerblue', linewidth=lw,
           label='Lasso')
  plt.plot(coef_multi_task_lasso_[:, feature_to_plot], color='gold', linewidth=lw,
           label='MultiTaskLasso')
  plt.legend(loc='upper center')
  plt.axis('tight')
  plt.ylim([-1.1, 1.1])
  plt.show()


if __name__=='__main__':
  #test()
  #result_merger()
  path = 'C:/Users/CNM/Downloads/fbcsp_example/MultiData/'
  files = os.listdir(path + '/CSP')
  #files = os.listdir(path)
  for file in files:
    #fbcsp_batch(file, path)
    #deep_learning_batch(file, path)
    #tdp_batch(file, path)
    feature_merging(file, ['CSP', 'PSD', 'csp', 'psdv'], path)
    #csp_batch(file, path)
    print('abc')
