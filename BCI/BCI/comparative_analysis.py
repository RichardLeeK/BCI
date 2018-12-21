import os, scipy.io
from sklearn import svm, linear_model
from sklearn import ensemble
import xgboost as xg
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score

def competition():
  path = 'E:/Richard/MSource'

def binary_RA(mode='tw'):
  os.chdir('E:/Richard/RA/2c_f/')
  files = os.listdir('csp')
  for file in files:
    try:
      csp = scipy.io.loadmat('csp/' + file)['csp_' + mode][0][0]
      tdp = scipy.io.loadmat('tdp/' + file)['tdp_' + mode][0][0]
      psd = scipy.io.loadmat('psd/' + file)['psd_' + mode][0][0]
      wve = scipy.io.loadmat('wve/' + file)['wve_' + mode][0][0]
    except:
      continue
    for i in range(5):
      #clfns = ['lsvm', 'ksvm', 'lda', 'srlda', 'gb', 'xgboost', 'mlp', 'rbm']
      clfns = ['mlp']
      fes = ['csp', 'tdp', 'psd', 'wve']
      for clfn in clfns:
        for fe in fes:
          if fe == 'csp':
            tx = np.transpose(csp[0][i]);
            ty = np.transpose(csp[1][i]).argmax(axis=1);
            vx = np.transpose(csp[2][i]);
            vy = np.transpose(csp[3][i]).argmax(axis=1);
          elif fe == 'tdp':
            tx = np.transpose(tdp[0][i]);
            ty = np.transpose(tdp[1][i]).argmax(axis=1);
            vx = np.transpose(tdp[2][i]);
            vy = np.transpose(tdp[3][i]).argmax(axis=1);
          elif fe == 'psd':
            tx = np.transpose(psd[0][i]);
            ty = np.transpose(psd[1][i]).argmax(axis=1);
            vx = np.transpose(psd[2][i]);
            vy = np.transpose(psd[3][i]).argmax(axis=1);
          elif fe == 'wve':
            tx = np.transpose(wve[0][i]);
            ty = np.transpose(wve[1][i]).argmax(axis=1);
            vx = np.transpose(wve[2][i]);
            vy = np.transpose(wve[3][i]).argmax(axis=1);

          if clfn == 'lsvm': clf = svm.LinearSVC()
          elif clfn == 'ksvm': clf = svm.SVC(kernel='linear')
          elif clfn == 'lda': clf = LinearDiscriminantAnalysis()
          elif clfn == 'srlda': lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
          elif clfn == 'gb': clf = ensemble.GradientBoostingClassifier()
          elif clfn == 'xgboost': clf = xg.XGBClassifier()
          elif clfn == 'mlp': clf = MLPClassifier();
          elif clfn == 'rbm':
            clf = Pipeline(steps=[('rbm', BernoulliRBM()), ('logistic', linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial'))])
          import multi
          tx, ty = multi.smote_application(tx, ty)
          clf.fit(tx, ty)
          y_predict = clf.predict(vx)
          coh = cohen_kappa_score(vy, y_predict)
          acc = accuracy_score(vy, y_predict)
          pen = open('res/' + mode + '/res_' + fe + '_' + clfn + '-smt.csv', 'a')
          pen.write(file + ',' + str(i) + ',' + str(coh) + ',' + str(acc) + '\n')
          pen.close()

def flatten_for_FBCSP(one):
  new = []
  for i in range(len(one)):
    for v in one[i]:
      new.append(v)
  return np.array(new)

def FBCSP_test():
  path = 'E:/Richard/MSource/Moduleing/Data/FBCSP/'
  files = os.listdir(path)
  for file in files:
    data = scipy.io.loadmat(path + file)['csp'][0][0]
    train_x = data[0]; train_y = data[1]
    test_x = data[2]; test_y = data[3]
    for i in range(5):
      tx = np.transpose(flatten_for_FBCSP(train_x[i]))
      ty = np.transpose(train_y[i]).argmax(axis=1)
      vx = np.transpose(flatten_for_FBCSP(test_x[i]))
      vy = np.transpose(test_y[i]).argmax(axis=1)
      clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      clf.fit(tx, ty)
      prd = clf.predict(vx)
      acc = accuracy_score(vy, prd)
      kap = cohen_kappa_score(vy, prd)
      sen = file + ',' + str(i) + ',' + str(acc) + ',' + str(kap) + '\n'
      pen = open('rf_comparative/FBCSP.csv', 'a')
      pen.write(sen)
      pen.close()

def CSSSP_test():
  path = 'E:/Richard/MSource/Moduleing/Data/CSSSP_f_bin/'
  files = os.listdir(path)
  for file in files:
    data = scipy.io.loadmat(path + file)['csp'][0][0]
    train_x = data[0]; train_y = data[1]
    test_x = data[2]; test_y = data[3]
    for i in range(5):
      tx = np.transpose(train_x[i])
      ty = np.transpose(train_y[i]).argmax(axis=1)
      vx = np.transpose(test_x[i])
      vy = np.transpose(test_y[i]).argmax(axis=1)
      clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      clf.fit(tx, ty)
      prd = clf.predict(vx)
      acc = accuracy_score(vy, prd)
      kap = cohen_kappa_score(vy, prd)
      sen = file + ',' + str(i) + ',' + str(acc) + ',' + str(kap) + '\n'
      pen = open('rf_comparative/csssp.csv', 'a')
      pen.write(sen)
      pen.close()

def test():
  data = scipy.io.loadmat('E:/Richard/MSource/1119/td/1_tw.mat')['csp_tw'][0][0]
  train_x = data[0]; train_y = data[1]
  test_x = data[2]; test_y = data[3]
  pen = open('res_wcsp.csv', 'a')
  for i in range(5):
    tx = np.transpose(train_x[i])
    ty = np.transpose(train_y[i]).argmax(axis=1)
    vx = np.transpose(test_x[i])
    vy = np.transpose(test_y[i]).argmax(axis=1)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    sentence = '1,' + str(i)
    for j in range(4):
      clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      clf.fit(tx[:,:,j], ty)
      pred = clf.predict(vx[:,:,j])
      coh = cohen_kappa_score(vy, pred)
      acc = accuracy_score(vy, pred)
      print(acc)
      sentence += ',' + str(acc)
    #sentence += ',' + 



if __name__ == '__main__':
  CSSSP_test()