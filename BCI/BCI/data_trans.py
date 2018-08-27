import scipy.io, CSP, BCI, pickle
import numpy as np
import RCNN


def one_fbcsp_for_kist(x, y):
  file = [{'clab':[], 'fs':250, 'x': [], 'y': []}, {'clab':[], 'fs':250, 'x': [], 'y': []}, {'clab':[], 'fs':250, 'x': [], 'y': []}]
  file[0]['clab'] = ['0', '12']
  file[1]['clab'] = ['1', '02']
  file[2]['clab'] = ['2', '01']
  for i in range(len(y)):
    if y[i] == 0:
      file[0]['x'].append(x[i]); file[0]['y'].append([0, 1])
      file[1]['x'].append(x[i]); file[1]['y'].append([1, 0])
      file[2]['x'].append(x[i]); file[2]['y'].append([1, 0])
    elif y[i] == 1:
      file[0]['x'].append(x[i]); file[0]['y'].append([1, 0])
      file[1]['x'].append(x[i]); file[1]['y'].append([0, 1])
      file[2]['x'].append(x[i]); file[2]['y'].append([1, 0])
    elif y[i] == 2:
      file[0]['x'].append(x[i]); file[0]['y'].append([1, 0])
      file[1]['x'].append(x[i]); file[1]['y'].append([1, 0])
      file[2]['x'].append(x[i]); file[2]['y'].append([0, 1])
  for v in file:
    v['x'] = np.transpose(np.array(v['x']))
    v['y'] = np.transpose(np.array(v['y']))
  return file


def load_kist_data(sub = '3'):
  import pickle
  print(sub)
  x = scipy.io.loadmat('kist_data/grasp/x_' + sub + '.mat')['x_' + sub].transpose()
  y = scipy.io.loadmat('kist_data/grasp/y_' + sub + '.mat')['y_' + sub].transpose().argmax(axis=1)

  y_ = np.array(BCI.lab_inv_translator(y))

  kv = BCI.gen_kv_idx(y_, 5)
  k = 1
  for train_idx, test_idx in kv:
    x_train, y_train = x[train_idx], y_[train_idx]
    x_test, y_test = x[test_idx], y_[test_idx]
    file = open('kist_data/grasp/np/' + sub + '_' + str(k) + '.pic', 'wb')
    pickle.dump({'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}, file)
    file.close()
    step = int(len(x[0][0]) / 5)
    for i in range(0, 5):
      cur_x_train = x_train[:, :, step * i: step * (i + 1)]
      cur_x_test = x_test[:, :, step * i: step * (i + 1)]
      for j in range(1, 10):
        cur_cur_x_train = CSP.arr_bandpass_filter(cur_x_train, j*4, (j+1)*4, 250)
        cur_cur_x_test = CSP.arr_bandpass_filter(cur_x_test, j*4, (j+1)*4, 250)

        f_train = one_fbcsp_for_kist(cur_cur_x_train, y_train.argmax(axis=1))
        f_test = one_fbcsp_for_kist(cur_cur_x_test, y_test.argmax(axis=1))

        scipy.io.savemat('mat/kist_ori/grasp/A0' + sub + 'T_' + str(i) + '_' + str(j) + '_1_' + str(k) + '_train.mat', f_train[0])
        scipy.io.savemat('mat/kist_ori/grasp/A0' + sub + 'T_' + str(i) + '_' + str(j) + '_2_' + str(k) + '_train.mat', f_train[1])
        scipy.io.savemat('mat/kist_ori/grasp/A0' + sub + 'T_' + str(i) + '_' + str(j) + '_3_' + str(k) + '_train.mat', f_train[2])

        scipy.io.savemat('mat/kist_ori/grasp/A0' + sub + 'T_' + str(i) + '_' + str(j) + '_1_' + str(k) + '_test.mat', f_test[0])
        scipy.io.savemat('mat/kist_ori/grasp/A0' + sub + 'T_' + str(i) + '_' + str(j) + '_2_' + str(k) + '_test.mat', f_test[1])
        scipy.io.savemat('mat/kist_ori/grasp/A0' + sub + 'T_' + str(i) + '_' + str(j) + '_3_' + str(k) + '_test.mat', f_test[2])

    k += 1

def x_translator(x):
  new_x = []
  for i in range(len(x[0])):
    new_x.append(x[:,i,:])

  return np.reshape(np.array(new_x), (len(new_x), len(new_x[0]), len(new_x[0][0]), 1))

def load_tscsp(sub = '3', epoch = 10):
  x = x_translator(scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/kist_rev/twist/A0' + sub + 'T.mat')['csp_2'])
  y = scipy.io.loadmat('kist_data/twist/y_' + sub + '.mat')['y_' + sub].transpose().argmax(axis=1)
  import RCNN

  y = np.array(BCI.lab_inv_translator(y))
  kv = BCI.gen_kv_idx(y, 2)
  acc = []; loss = [];
  for train_idx, test_idx in kv:
    x_train, y_train = x[train_idx], y[train_idx]
    x_train = x_train.reshape(len(x_train), 48, 45, 1)
    x_test, y_test = x[test_idx], y[test_idx]
    x_test = x_test.reshape(len(x_test), 48, 45, 1)
    model = RCNN.create_model((48, 45, 1))
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, batch_size=10)
    metrics = model.evaluate(x_test, y_test)
    pen = open('rcnn_res_5.csv', 'a')
    pen.write('RCNN,' + sub + ',' + str(epoch) + ',' + str(metrics[1]) + '\n')
    pen.close()


def pca_gen(mode = 'twist', sub = '3', epoch = 10):
  from sklearn.decomposition import PCA
  import matplotlib.pyplot as plt
  for fold in range(1, 6):
    x_train = x_translator(scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/kist_rev/' + mode + '/A0' + sub + 'T_' + str(fold) + '_train.mat')['csp_2_train'])
    x_test = x_translator(scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/kist_rev/A0' + sub + 'T_' + str(fold) + '_test.mat')['csp_2_test'])
    file = open('kist_data/'+ mode +'/np/' + sub + '_' + str(fold) + '.pic', 'rb')
    raw = pickle.load(file)
    file.close()
    y_train = raw['y_train']
    y_test = raw['y_test']
    x_train = np.reshape(x_train, (len(x_train), 45 * 48))
    pca = PCA(n_components=2)
    X_r = pca.fit(x_train).transform(x_train)
    y = y_train.argmax(axis = 1)
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    target_names = ['1', '2', '3']
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
      plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=2,
                label=target_name)
    plt.savefig('lasdjkflaksdjflsadkjflaskdjf.eps', format='eps', dpi=1000)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    #clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf = SVC()
    clf.fit(x_train, y_train.argmax(axis=1))
    accuracy = clf.score(x_test, y_test.argmax(axis=1))
    pen = open('Opt_' + mode + '.csv', 'a')
    pen.write('Opt' + ',' + sub + ',' + str(fold) + ',' + str(accuracy) + '\n')



def load_ts_origin_twist(sub = '3'):
  for fold in range(1, 6):
    x_train = x_translator(scipy.io.loadmat())



def load_ts_rev(sub = '3'):
  for fold in range(1, 6):
    x_train = x_translator(scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/kist_rev/A0' + sub + 'T_' + str(fold) + '_train.mat')['csp_2_train'])
    file = open('kist_data/np/' + sub + '_' + str(fold) + '.pic', 'rb')
    raw = pickle.load(file)
    file.close()
    y_train = raw['y_train']
    
    x_test = x_translator(scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/kist_rev/A0' + sub + 'T_' + str(fold) + '_test.mat')['csp_2_test'])
    y_test = raw['y_test']

    #x_train = np.reshape(x_train, (len(x_train), 48, 45))
    #x_test = np.reshape(x_test, (len(x_test), 48, 45))
    import CNN
    model = CNN.create_model((48, 45, 1))
    #model = RCNN.create_model((48, 45, 1))
    #model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size = 5)
    #metrics = model.evaluate(x_test, y_test)
    model.fit(x_test, y_test, validation_data=(x_train, y_train), epochs=100, batch_size = 5)
    metrics = model.evaluate(x_train, y_train)
    pen = open('rcnn_res!!!!!_new.csv', 'a')
    pen.write('RCNN,' + sub + ',' + str(fold) + ',' + str(metrics[1]) + '\n')
    pen.close()
    print('abc')



def optimal_ts_batch(mode = 'twist', sub = '1'):
  for fold in range(1, 6):
    x_train = scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/kist_rev/' + mode + '/A0' + sub + 'T_' + str(fold) + '_train.mat')['csp_2_train']
    x_test = scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/kist_rev/A0' + sub + 'T_' + str(fold) + '_test.mat')['csp_2_test']
    file = open('kist_data/'+ mode +'/np/' + sub + '_' + str(fold) + '.pic', 'rb')
    raw = pickle.load(file)
    file.close()
    y_train = raw['y_train']
    y_test = raw['y_test']
    import feature_selection as FS
    opt_idx = FS.lsvm_filter_pp2(x_train, y_train)
    x_train = np.transpose(x_train[:,:,opt_idx])
    x_test = np.transpose(x_test[:,:,opt_idx])
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    #clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf = SVC()
    clf.fit(x_train, y_train.argmax(axis=1))
    accuracy = clf.score(x_test, y_test.argmax(axis=1))
    pen = open('Opt_' + mode + '.csv', 'a')
    pen.write('Opt' + ',' + sub + ',' + str(fold) + ',' + str(accuracy) + '\n')

def rcnn_ts_batch(mode='twist', sub = '1'):
  for fold in range(1, 6):
    x_train = x_translator(scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/kist_rev/' + mode + '/A0' + sub + 'T_' + str(fold) + '_train.mat')['csp_2_train'])
    x_test = x_translator(scipy.io.loadmat('F:/KIST/source/BCI/BCI/BCI/mat/kist_rev/A0' + sub + 'T_' + str(fold) + '_test.mat')['csp_2_test'])
    file = open('kist_data/'+ mode +'/np/' + sub + '_' + str(fold) + '.pic', 'rb')
    raw = pickle.load(file)
    file.close()
    y_train = raw['y_train']
    y_test = raw['y_test']
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    #clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf = SVC()
    clf.fit(x_train, y_train.argmax(axis=1))
    accuracy = clf.score(x_test, y_test.argmax(axis=1))
    pen = open('Opt_' + mode + '.csv', 'a')
    pen.write('Opt' + ',' + sub + ',' + str(fold) + ',' + str(accuracy) + '\n')


if __name__ == '__main__':
  for i in range(3, 16):
    #load_kist_data(str(i))
    #load_tscsp(str(i), 10)
    #optimal_ts_batch('twist', str(i))
    #load_ts_rev(str(i))
    pca_gen()