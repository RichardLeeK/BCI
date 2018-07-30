from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
import numpy as np
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adam

def create_model(input_shape):
  model = Sequential()
  model.add(Conv1D(16, input_shape=(3000, 64), kernel_size=(9)))
  model.add(Conv1D(16, kernel_size=(9)))
  model.add(Conv1D(16, kernel_size=(9)))
  model.add(Dropout(0.5))
  model.add(MaxPooling1D(pool_size=(4)))
  model.add(Conv1D(32, kernel_size=(1)))
  model.add(Conv1D(32, kernel_size=(9)))
  model.add(Conv1D(32, kernel_size=(9)))
  model.add(Conv1D(32, kernel_size=(9)))
  model.add(Dropout(0.5))
  model.add(MaxPooling1D(pool_size=(4)))
  model.add(Conv1D(64, kernel_size=(1)))
  model.add(Conv1D(64, kernel_size=(9)))
  model.add(Conv1D(64, kernel_size=(9)))
  model.add(Conv1D(64, kernel_size=(9)))
  model.add(Dropout(0.5))
  model.add(MaxPooling1D(pool_size=(4)))
  model.add(Conv1D(128, kernel_size=(22)))
  model.add(Conv1D(256, kernel_size=(1)))
  model.add(Dense(512))
  model.add(Dropout(0.5))
  model.add(Dense(64))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(2, activation='softmax'))
  #opt = RMSprop(lr=0.001, decay=1e-6)
  #opt = SGD(lr=0.001, decay=1e-6, momentum=0.9)
  opt = Adam(decay=1e-6)
  model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

  print(model.summary())
  
  
  return model

def batch():
  import json
  with open('test.js', 'r') as f:
    data = json.load(f)
  import BCI
  for k in data:
    x = np.array(data[k]['x'])
    y = data[k]['y']
    y = np.array(BCI.lab_inv_translator(y))
    kv = BCI.gen_kv_idx(y, 10)
    acc = []; loss = [];
    for train_idx, test_idx in kv:
      x_train, y_train = x[train_idx], y[train_idx]
      x_test, y_test = x[test_idx], y[test_idx]
      model = create_model((1, 1, 1))
      model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)
      print('abc')




def temp_x_y(x, y):
  x_ = []; y_ = [];
  for i in range(len(y)):
    if y[i] == 1.0:
      x_.append(x[i])
      y_.append([1, 0])
    elif y[i] == 2.0:
      x_.append(x[i])
      y_.append([0, 1])
    
  return np.array(x_), np.array(y_)

def resting_classification(epoch):
  import json
  with open('test2.js', 'r') as f:
    data = json.load(f)
  import BCI
  for k in data:
    x = np.array(data[k]['x'])
    y = data[k]['y']
    y = np.array(BCI.inv_translator_for_resting(y))
    x, y = temp_x_y(x, y)
    kv = BCI.gen_kv_idx(y, 10)
    acc = []; loss = [];
    for train_idx, test_idx in kv:
      x_train, y_train = x[train_idx], y[train_idx]
      x_test, y_test = x[test_idx], y[test_idx]
      try:
        model = create_model((1, 1, 1))
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, batch_size=16)
        metrics = model.evaluate(x_test, y_test)
        acc.append(metrics[1])
        loss.append(metrics[0])
      except Exception as e:
        pen = open('rstnn.csv', 'a')
        pen.write('RSTNN_LR,' + k + ',' + str(e) + '\n')
        pen.close()
      if len(acc) == 0: continue
      pen = open('rstnn.csv', 'a')
      pen.write('RSTNN_LR,' + k + ',' + str(epoch) + ',' + str((sum(acc)+0.00001) / float(len(acc))) + '\n')
      pen.close()

def load_raw_data(path):
  import os
  files = os.listdir(path)
  data = {}
  for file in files:
    print(file)
    f = open(path + '/' + file)
    lines = f.readlines()
    cur_x = []
    for line in lines:
      sl = line.split(',')
      cur_x_one_line = []
      for v in sl:
        cur_x_one_line.append(float(v))
      cur_x.append(cur_x_one_line)
    sf = file.split('_')
    if sf[1] not in data:
      data[sf[1]] = {}
    if 'x' not in data[sf[1]]:
      data[sf[1]]['x'] = []
    if 'y' not in data[sf[1]]:
      data[sf[1]]['y'] = []
    data[sf[1]]['x'].append(cur_x)
    data[sf[1]]['y'].append(float(sf[3].split('.')[0]))
  import json
  with open('test2.js', 'w') as f:
    json.dump(data, f)
  print('fin')

if __name__ == '__main__':
  for i in range(1, 10):
    resting_classification(i * 10)
  #load_raw_data('G:/Richard/작업공간/KIST 로봇팔/Source code/dat/test')


