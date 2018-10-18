import numpy as np
from sklearn.svm import SVC, SVR
import BCI, CSP, BCI, RCNN_2nd
from sklearn.feature_selection import RFE
import scipy.io, os
def parzen_kde(train, test, window):
  from scipy.stats import multivariate_normal
  train_size = np.shape(train)[0]
  test_size = np.shape(test)[0]
  #num_feature = len(train[1])
  num_feature = 1
  covariance = np.zeros((num_feature, num_feature))
  for i in range(num_feature):
    covariance[i][i] = np.var(train)
  estimated_density = np.zeros((test_size, 1))
  for i in range(len(test)):
    x = test[i]
    test_sample_matrix = np.ones((train_size, 1)) * x
    new_diff = test_sample_matrix - np.reshape(train, (len(train), 1))
    for j in range(num_feature):
      new_diff[abs(new_diff[:, j]) > window, j] = 10000000
    mvn = multivariate_normal(np.zeros((1, num_feature)), covariance)
    estimated_density[i] = np.mean((1/(window**num_feature)) * mvn.pdf((new_diff/window)))
  return estimated_density
def mutual_information(f1, f2):
  merge = np.concatenate((f1, f2), axis=0)
  estimated_density = parzen_kde(merge, merge, 1)
  entropy = -np.sum(np.log(estimated_density)) / len(estimated_density)

  class_one_density = parzen_kde(f1, f1, 1)
  class_two_density = parzen_kde(f2, f2, 1)

  hac_one = -np.sum(np.log(class_one_density)) / len(class_one_density)
  hac_two = -np.sum(np.log(class_two_density)) / len(class_two_density)

  cond_entropy = (hac_one + hac_two) / 2
  return entropy - cond_entropy

def fb_mibif_with_csp(x, y, fb_csp):
  xs = [[], [], [], []]
  for i in range(np.shape(fb_csp)[1]):
    xs[y[i]].append(np.transpose(fb_csp[:,i,:]))
  mis = np.zeros((len(xs), len(xs[0]) + 10, len(xs[0][0])))
  for i in range(len(xs)): # class number
    print('i: ' + str(i))
    for j in range(len(xs[i])): # epoch count
      for k in range(len(xs[i][j])): # filter number
        one = xs[i][j][k]
        rest = []
        for l in range(len(xs)):
          if i == l: continue
          try: rest.extend(xs[l][j][k])
          except: continue
        mis[i][j][k] = mutual_information(one, np.array(rest))
  return np.sum(np.sum(mis, axis=1), axis=0).argmax()

def lsvm_filter(x, y):
  #x = np.array(x)[:,:,18:]
  scores = np.zeros(x.shape[-1])

  kv = BCI.gen_kv_idx(BCI.lab_inv_translator(y, 5))
  for train_idx, test_idx in kv:
    for i in range(x.shape[-1]):
      x_train, y_train = x[train_idx,:,i], y[train_idx]
      x_test, y_test = x[test_idx,:,i], y[test_idx]
      clf = SVC(kernel='linear')
      clf.fit(x_train, y_train)
      scores[i] += clf.score(x_test, y_test)
  return np.array(scores).argmax()
    
def all_features(x):
  new_x = []
  for i in range(x.shape[0]):
    new_x.append(x[i,:,:].flatten())
  return np.array(new_x)
  



  

def lsvm_filter_pp(x1, y1, x2, y2):
  scores = []
  for i in range(len(x1[0][0])):
    train_x = np.transpose(np.array(x1[:,:,i]))
    valid_x = np.transpose(np.array(x2[:,:,i]))
    clf = SVC(kernel='linear')
    clf.fit(train_x, y1.argmax(axis=1))
    scores.append(clf.score(valid_x, y2.argmax(axis=1)))
  return np.array(scores).argmax()

def lsvm_filter_pp2(x, y):
  scores = []
  for i in range(len(x[0][0])):
    kv = BCI.gen_kv_idx(y, 9)
    cur_scores = []
    for train_idx, test_idx in kv:
      train_x, train_y = x[:,train_idx,:], y[train_idx]
      test_x, test_y = x[:,test_idx,:], y[test_idx]
      train_x = np.transpose(np.array(train_x[:,:,i]))
      test_x = np.transpose(np.array(test_x[:,:,i]))
      clf = SVC(kernel='linear')
      clf.fit(train_x, train_y.argmax(axis=1))
      cur_scores.append(clf.score(test_x, test_y.argmax(axis=1)))
    scores.append(np.mean(cur_scores))
  return np.array(scores).argmax()

def mibif_filter(x, y):
  #y_ = y.argmax(axis=1)
  xs = [[], [], [], [], []]
  for i in range(x.shape[0]):
    xs[y[i]].append(x[i,:,:])
  mis = np.zeros((len(xs), len(xs[0]), len(xs[0][0][0])))
  for i in range(len(xs)): # class number
    for j in range(len(xs[i])): # epoch count
      for k in range(len(xs[i][j][0])): # filter number
        one = xs[i][j][:,k]
        rest = []
        for l in range(len(xs)):
          if i == l: continue
          try: rest.extend(xs[l][j][:,k])
          except: continue
        try:
          mis[i][j][k] = mutual_information(one, np.array(rest))
        except:
          break
  return np.sum(np.sum(mis, axis=1), axis=0).argmax()



def lsvm_wrapper(x, y):
  max_csp = np.zeros((x.shape[2]))
  for i in range(x.shape[2]):
    result_feature=np.zeros((1, x.shape[0]))
    svm = SVR(kernel='linear')
    rfe = RFE(estimator=svm, n_features_to_select=1, step=1)
    rfe = rfe.fit(x[:,:,i], y)
    
    result_feature =np.transpose(rfe.transform(x[:,:,i]))
    max_csp[i] = np.max(result_feature)
  return np.argmax(max_csp)

def classifier_svm(train_x, train_y, test_x, test_y, i, fold):
  clf = SVC(decision_function_shape='ovo')
  train_y = np.transpose(train_y).argmax(axis=1)
  clf.fit(np.transpose(train_x), train_y)
  predict = clf.predict(np.transpose(test_x))
  test_y = np.transpose(test_y).argmax(axis=1)
  score = np.sum(np.equal(predict, test_y))/len(predict)
  
  pen = open('result.csv', 'a')
  pen.write('SVM,' + str(i) + ',' + str(fold) + ','+ str(score) + '\n')
  pen.close()

  print(str(i)+"_"+str(fold)+"  "+str(score))
  

def classifier_rlda(train_x, train_y, test_x, test_y, i, fold):
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  rlda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
  train_x = np.transpose(train_x)
  test_x= np.transpose(test_x)
  rlda.fit(train_x, np.transpose(train_y).argmax(axis=1))
  fb_score = rlda.score(test_x, np.transpose(test_y).argmax(axis=1))

  pen = open('result.csv', 'a')
  pen.write('RLDA,' + str(i) + ',' + str(fold) + ','+ str(fb_score) + '\n')
  pen.close()

  print(str(i)+"_"+str(fold)+"  "+str(fb_score))


if __name__ == "__main__":
  data_dir = 'data/csp_data'

  for i in range(1, 14):
    for fold in range(1, 6):
      train_data = scipy.io.loadmat(os.path.join(data_dir, str(i)+'_'+str(fold)+'_train.mat'))
      test_data = scipy.io.loadmat(os.path.join(data_dir, str(i)+'_'+str(fold)+'_test.mat'))

      train_x = train_data['train'][0][0][0]
      train_y = train_data['train'][0][0][1]
      test_x = test_data['test'][0][0][0]
      test_y = test_data['test'][0][0][1]
      feature = csp_svm(train_x, train_y)
      #classifier_svm(train_x[feature, :, :], train_y, test_x[feature, :, :], test_y, i, fold)
      classifier_rlda(train_x[feature, :, :], train_y, test_x[feature, :, :], test_y, i, fold)

      del train_x, train_y, test_x, test_y, feature

#target_path = 'D:/Innea/STUDY/BCI-robotarm/data/twist/subdata'
#input_path = 'D:/Innea/STUDY/BCI-robotarm/feature extraction wrapper/data'
#for name in os.listdir(target_path):
#  name = name.split('.')[0]
  #make_data(name)
  #svm_generation(name)

  

  #target_name = os.path.join(target_path, name)
  #mat = scipy.io.loadmat(target_path)
  