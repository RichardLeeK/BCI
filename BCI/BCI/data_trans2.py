
import scipy.io, CSP, BCI, pickle
import numpy as np
import RCNN
from sklearn.decomposition import PCA

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

if __name__ == '__main__':
  for i in range(3, 16):
#    load_kist_data(str(i))

    rcnn_ts_batch(str(i))