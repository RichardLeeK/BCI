import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout

from sklearn import svm
from sklearn.model_selection import KFold

import os

import RCNN

from keras.callbacks import EarlyStopping

csp_val = 'raw'
shape = (5, 4, 6)

def loader(file, mode):
  fx = open(file + '_' + mode + '__' + str(csp_val) + '_x.csv')
  lines = fx.readlines()
  fx.close()

  x = []
  for line in lines:
    sl = line.split(',')
    cur_x = []
    for val in sl:
      cur_x.append(float(val))
    x.append(cur_x)
  x = np.array(x)
  x = np.transpose(x)

  fy = open(file + '_' + mode + '__' + str(csp_val) + '_y.csv')
  lines = fy.readlines()
  fy.close()

  y = []
  for line in lines:
    sl = line.split(',')
    cur_y = []
    for val in sl:
      cur_y.append(float(val))
    y.append(cur_y)
  y = np.array(y)
  y = np.transpose(y)
  return x, y

def create_bDNN():
  model = Sequential()
  model.add(Dense(6, input_dim=12, activation='relu'))
  model.add(Dense(6, activation='relu'))
  model.add(Dense(4, activation='relu'))
  model.add(Dense(4, activation='relu'))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
  return model

def create_mDNN():
  model = Sequential()
  model.add(Dense(24, input_dim=36, activation='relu'))
  model.add(Dense(24, activation='relu'))
  model.add(Dense(12, activation='relu'))
  model.add(Dense(12, activation='relu'))
  model.add(Dense(6, activation='softmax'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
  return model

def create_tDNN():
  model = Sequential()
  model.add(Dense(12, input_dim=18, activation='relu'))
  model.add(Dense(12, activation='relu'))
  model.add(Dense(6, activation='relu'))
  model.add(Dense(6, activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(3, activation='softmax'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
  return model


def create_bCNN():
  model = Sequential()
  model.add(Convolution1D(nb_filter=512, filter_length=1, input_shape=(12, 1)))
  model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dropout(0.4))
  model.add(Dense(2))
  model.add(Activation('softmax'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
  return model

def create_tRCNN():
  model = RCNN.makeModel(18, 6, 3, 3)
  model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'])
  return model


def create_SVM():
  clf = svm.SVC()
  return clf

def get_file_name(file):
  return file.split('_')[0]

def transition_y(y):
  n_y = []
  for i in range(0, len(y)):
    n_y.append(y[i][0])
  return n_y


def one_routine(path, file, mode):
  os.chdir(path)
  nf = get_file_name(file)
  x, y = loader(nf, mode)
  kf = KFold(n_splits=20, shuffle=True)
  kv = kf.split(x)
  acc = []
  for train_idx, test_idx in kv:
    x_train, x_test = x[train_idx], x[test_idx]
    x_train = x_train.transpose()[:1800].transpose()
    x_test = x_test.transpose()[:1800].transpose()
    y_train, y_test = y[train_idx], y[test_idx]
    DNN = RCNN.create_model(csp_val=csp_val)
    x_train = x_train.reshape((x_train.shape[0], 100, 6, 3))
    x_test = x_test.reshape((x_test.shape[0], 100, 6, 3))

    epoch = 10000
    es = EarlyStopping(monitor='val_acc', patience=500, mode='auto')
    DNN.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, callbacks=[es], batch_size=1)
    metrics = DNN.evaluate(x_test, y_test)
    
    for i in range(len(DNN.metrics_names)):
      if (str(DNN.metrics_names[i]) == 'acc'):
        acc.append(metrics[i])
      if (str(DNN.metrics_names[i]) == 'categorical_accuracy'):
        acc.append(metrics[i])
      print(str(DNN.metrics_names[i]) + ": " + str(metrics[i]))
  pen = open('../result_S2_'+mode+'.' + str(csp_val) + '.csv', 'a')
  pen.write(nf + ',' + mode + ',RCNN_raw_1,' + str(epoch) + ',' + str(sum(acc) / float(len(acc)))+'\n')
  pen.close()

  pen2 = open('../result_S2_detailv_'+mode+'.' + str(csp_val) + '.csv', 'a')
  for accs in acc:
    pen2.write(nf + ',' + mode + ',RCNN_raw_1,' + str(epoch) + ',' + str(acc)+'\n')
  pen2.close()