import os, scipy.io

def batch():
  path = 'E:/Richard/TSOSPData/grasp/'
  files = os.listdir(path)
  for file in files:
    if file.split('-')[-1] == 'test.mat':
      continue
    test_file = file.split('-')[0] + '-test.mat'
    train_file = file

    train = scipy.io.loadmat(path + train_file)['total_train'][0]
    test = scipy.io.loadmat(path + test_file)['total_test'][0]

    for tem in range(0, 5):
      tx = train[0][0][0][0]
      ty = train[0][0][0][1]

      vx = test[0][0][0][0]
      vy = test[0][0][0][1]


    print('abc')


if __name__ == '__main__':
  batch()

