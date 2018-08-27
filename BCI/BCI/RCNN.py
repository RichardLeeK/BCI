from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.layers import Input, add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
import numpy as np

def RCL_block(l_settings, l, pool=True):
    input_num_filters = l_settings.output_shape[1]
    conv1 = Conv2D(input_num_filters, (1, 1), padding='same')
    stack1 = conv1(l)   	
    stack2 = BatchNormalization()(stack1)
    stack3 = PReLU()(stack2)

    conv2 = Conv2D(input_num_filters, (3, 3), padding='same', kernel_initializer = 'he_normal')
    stack4 = conv2(stack3)
    #stack5 = merge([stack1, stack4], mode='sum')
    stack5 = add([stack1, stack4])
    stack6 = BatchNormalization()(stack5)
    stack7 = PReLU()(stack6)

    conv3 = Conv2D(input_num_filters, (3, 3), padding='same', weights = conv2.get_weights())
    stack8 = conv3(stack7)
    #stack9 = merge([stack1, stack8], mode='sum')
    stack9 = add([stack1, stack8])
    stack10 = BatchNormalization()(stack9)
    stack11 = PReLU()(stack10)    

    conv4 = Conv2D(input_num_filters, (3, 3), padding='same', weights = conv2.get_weights())
    stack12 = conv4(stack11)
    #stack13 = merge([stack1, stack12], mode='sum')
    stack13 = add([stack1, stack12])
    stack14 = BatchNormalization()(stack13)
    stack15 = PReLU()(stack14)
    if pool:
      stack15 = MaxPool2D((2, 2), padding='same')(stack15)
    #stack16 = stack15
    stack16 = Dropout(0.5)(stack15)
            
    return stack16





def create_model(resize, output_dim=3, nb_layer=2):
  input = Input(resize)
  conv = Conv2D(128, (3, 3), padding='same', activation='relu')
  l = conv(input)

  for i in range(nb_layer):
    if i % 2 == 0:
      l = RCL_block(conv, l, pool=False)
    else:
      l = RCL_block(conv, l, pool=True)
  out = Flatten()(l)
  out = Dense(3, activation='softmax')(out)
  model = Model(input = input, output = out)
  print(model.summary())
  model.compile(loss = 'categorical_crossentropy', optimizer = 'Adadelta', metrics = ['accuracy'])
  return model

def create_model2(resize, output_dim=3, nb_layer=4):
  input = Input(resize)
  conv = Conv2D(128, (3, 3), padding='same', activation='relu')
  l = conv(input)

  for i in range(nb_layer):
    if i % 2 == 0:
      l = RCL_block(conv, l, pool=False)
    else:
      l = RCL_block(conv, l, pool=True)
  l = Dropout(0.5)(l)
  out = Flatten()(l)
  out = Dense(4, activation='softmax')(out)
  model = Model(input = input, output = out)
  print(model.summary())
  model.compile(loss = 'categorical_crossentropy', optimizer = 'Adadelta', metrics = ['accuracy'])
  return model


def csv_to_pickle(path):
  import os
  files = os.listdir(path)
  data = {}
  for file in files:
    print(file)
    f = open(path + '/' + file)
    lines = f.readlines()
    f.close()
    x = []; y = [];
    for line in lines:
      cur_x = []
      sl = line.split(',')
      cur_y = int(sl[0])
      for v in sl[1:]:
        cur_x.append(float(v.strip()))
      x.append(cur_x); y.append(cur_y);
    data[file] = {}
    data[file]['x'] = x
    data[file]['y'] = y

  import pickle
  with open('sb_csp_60.pk', 'wb') as f:
    pickle.dump(data, f)
  print('fin')

def batch(epoch):
  import pickle
  with open('sb_csp_60.pk', 'rb') as f:
    data = pickle.load(f)
  import BCI
  for k in data:
    x = np.array(data[k]['x'])
    y = data[k]['y']
    y = np.array(BCI.lab_inv_translator(y))
    kv = BCI.gen_kv_idx(y, 20)
    acc = []; loss = [];
    for train_idx, test_idx in kv:
      x_train, y_train = x[train_idx], y[train_idx]
      x_train = x_train.reshape(len(x_train), 3, 6, 5)
      x_test, y_test = x[test_idx], y[test_idx]
      x_test = x_test.reshape(len(x_test), 3, 6, 5)
      
      model = create_model((3, 6, 5))
      model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch)
      metrics = model.evaluate(x_test, y_test)
      pen = open('rcnn_res_60.csv', 'a')
      pen.write('RCNN,' + k + ',' + str(epoch) + ',' + str(metrics[1]) + '\n')
      pen.close()
      
      
      


if __name__ == '__main__':
  """
  csv_to_pickle('G:\\Richard\\작업공간\\KIST 로봇팔\\Source code\\dat\\0730_RCNN\\')

  """


  for i in range(0, 100):
    batch(i * 10)
