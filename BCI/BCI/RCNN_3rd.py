import os, scipy.io, RCNN, CNN
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
def compare(file):
  try:
    data = scipy.io.loadmat(file)['csp_gp'][0][0]
  except:
    return
  train_x = data[0]; train_y = data[1]
  test_x = data[2]; test_y = data[3]
  for i in range(5):
    tx = np.transpose(train_x[i])
    ty = np.transpose(train_y[i]).argmax(axis=1)
    vx = np.transpose(test_x[i])
    vy = np.transpose(test_y[i]).argmax(axis=1)
    lda = LinearDiscriminantAnalysis()
    mlp = MLPClassifier()
    srlda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(tx, ty); lda_p = lda.predict(vx);
    mlp.fit(tx, ty); mlp_p = mlp.predict(vx);
    srlda.fit(tx, ty); srlda_p = srlda_p = srlda.predict(vx);
    sub = file.split('_')[-2]
    pen = open('compare.csv', 'a')
    pen.write(sub + ',' + str(accuracy_score(vy, lda_p)) + ',' + str(accuracy_score(vy, mlp_p)) + ',' + str(accuracy_score(vy, srlda_p)) + '\n')
    pen.close()

def classification(file):
  data = scipy.io.loadmat(file)['csp'][0][0]
  train_x = data[0]; train_y = data[1]
  test_x = data[2]; test_y = data[3]
  for i in range(5):
    tx = np.transpose(train_x[i])
    tx = np.reshape(tx, (tx.shape[0], tx.shape[1], tx.shape[2], 1))
    ty = np.transpose(train_y[i])
    vx = np.transpose(test_x[i])
    vx = np.reshape(vx, (vx.shape[0], vx.shape[1], vx.shape[2], 1))
    vy = np.transpose(test_y[i])
    from keras.callbacks import EarlyStopping
    model = RCNN.create_model((tx.shape[1], tx.shape[2], 1))
    #model = CNN.create_model((tx.shape[1], tx.shape[2], 1))
    #model.fit(tx[:60,:,:], ty[:60,:], validation_data=(vx, vy), epochs=100, shuffle = True)
    model.fit(tx, ty, validation_data=(vx, vy), epochs=100, shuffle = True)
    metrics = model.evaluate(vx, vy)
    pen = open('rcnn_1107.csv', 'a')
    sub = file.split('_')[-1].split('.')[0]
    pen.write('RCNN,' + sub + ',' + str(metrics[1]) + '\n')


if __name__ == '__main__':
  path = 'E:/Richard/3CData/CSP22/'
  #path = 'E:/Richard/RCNNData/3C/gvt/'
  files = os.listdir(path)
  for f in files:
    compare(path + f)