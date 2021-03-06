import os
from PIL import Image
import numpy as np
import itertools
from sklearn.externals import joblib
import RCNN
import BCI
import CNN

def image_matrix_processing(path):
  output_path = 'fig/mat/'
  files = os.listdir(path)
  dic = {}
  for file in files:
    print(file)
    f = open(path+file, 'r')
    line = f.readline()
    f.close()
    sl = line.split(',')
    cur_x = []
    for v in sl:
      cur_x.append(float(v))
    
    sf = file.split('_')
    if sf[0] not in dic:
      dic[sf[0]] = {}
    if sf[2] not in dic[sf[0]]:
      dic[sf[0]][sf[2]] = {'x': [[], [], [], [], [], []], 'y': int(sf[3])}
    idx = int(float(sf[4].split('s')[0]) * 2) - 1
    
    dic[sf[0]][sf[2]]['x'][idx] = np.array(cur_x)
  joblib.dump(dic, output_path + 'grasp_MI.pic')  

def test(path):
  img = Image.open(path+'/bwyu_twist_1_2_0.0s.png')
  img = img.crop()
  img.show()
  # out:(210,85,690,570)
  # midium: (250,125,650,530)
  # in: (280, 155, 620, 500)
def batch_crop(path, crop_size=(280, 155, 620, 500), re_size=(32, 32)):
  output_path = 'fig/abs/in/32/'
  sample_cnt = 3
  files = os.listdir(path)
  dic = {}
  for file in files:
    print(file)
    img = Image.open(path+file)
    img = img.crop(crop_size)
    img = img.resize((re_size))
    if sample_cnt > 0:
      img.save(output_path + 'sample_'+str(sample_cnt)+'.png', format='png')
      sample_cnt -= 1
    np_img = np.array(img.getdata())
    sf = file.split('_')
    if sf[0] not in dic:
      dic[sf[0]] = {}
    if sf[2] not in dic[sf[0]]:
      dic[sf[0]][sf[2]] = {'x': [[], [], [], [], [], []], 'y': int(sf[3])}
    idx = int(float(sf[4].split('s')[0]) * 2)
    
    dic[sf[0]][sf[2]]['x'][idx] = np_img
  joblib.dump(dic, output_path + 'dat.pic')  
  
def data_transform_for_RCNN(dat, resize):
  x = []
  y = []
  for k, v in dat.items():
    c_x = np.array(v['x']).reshape(resize, resize, 18)
    c_y = v['y']
    x.append(c_x)
    y.append(c_y)
  rev_y = []
  for l in y:
    cur = [0, 0, 0]
    cur[l - 1] = 1
    rev_y.append(cur)
  return np.array(x), np.array(rev_y)

def one_train_test(file, data, resize, epoch):
  print('a')
def data_transform_for_CNN(dat, resize):
  x = []
  y = []
  for k, v in dat.items():
    for l in v['x']:
      x.append(l.reshape(resize, resize, 3))
      y.append(v['y'])
  rev_y = []
  for l in y:
    cur = [0, 0, 0]
    cur[l - 1] = 1
    rev_y.append(cur)
  return np.array(x), np.array(rev_y)

def one_train_test(data, resize):
  x, y = data_transform_for_RCNN(data, resize)
  kv = BCI.gen_kv_idx(y, 10)
  acc = []; loss = []
  for train_idx, test_idx in kv:
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    model = RCNN.create_model(resize)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch)
    metrics = model.evaluate(x_test, y_test)
    for i in range(len(model.metrics_names)):
      if (str(model.metrics_names[i] == 'acc')):
        acc.append(metrics[i])
      if (str(model.metrics_names[i] == 'loss')):
        loss.append(metrics[i])

  pen = open('/result.csv', 'a')

  config_sen = file+',RCNN,out,'+str(resize)+','+str(epoch)+','

  acc_sen = ''
  for v in acc: acc_sen += (str(v) + ',')
  acc_sen += (str(sum(acc) / float(len(acc))) + ',')
  loss_sen = ''
  for v in loss: los_sen += (str(v) + ',')
  loss_sen += (str(sum(loss) / float(len(loss))) + ',')
  pen.write(config_sen + acc_sen + loss_sen + '\n')
  pen.close()

def train_test():
  resize = 64
  path = 'fig/ori/out/' + str(resize) + '/'
  #model = CNN.create_model()
  #
  full_dat = joblib.load(path + 'dat.pic')
  #one_train_test(full_dat['bwyu'], resize)
  
  data_transform_for_CNN(full_dat['bwyu'], resize)

def train_test_3DCNN(data, resize):
  x, y = data_transform_for_RCNN(data, resize)
  kv = BCI.gen_kv_idx(y, 5)
  acc = []
  for train_idx, test_idx in kv:
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    model = CNN.create_3d_model()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
    metrics = model.evaluate(x_test, y_test)
    for i in range(len(model.metrics_names)):
      if (str(model.metrics_names[i] == 'acc')):
        acc.append(metrics[i])
  pen = open('/result.csv', 'a')
  acc_sen += [str(v) for v in acc]
  pen.write()


def test(sub = '1'):
  import scipy.interpolate
  import numpy
  import matplotlib 
  import matplotlib.pyplot as plt
  import Competition
  path='data/A0' + sub + 'T.npz'
  x, y = Competition.load_one_data(path)


  # close old plots
  plt.close("all")

  # some parameters
  N = 300             # number of points for interpolation
  xy_center = [2,2]   # center of the plot
  radius = 2          # radius

  # mostly original code
  """
  meanR = [9.95184937,   9.87947708,   9.87628496,   9.78414422,
           9.79365258,   9.96168969,   9.87537519,   9.74536093,
          10.16686878,  10.04425475,  10.10444126,  10.2917172 ,
          10.16745917,  100.0235203 ,   9.89914   ,  10.11263505,
           9.99756449,  10.17861254,  10.04704248]
  
  
  koord = [[1,4],[3,4],[1,3],[3,3],[2,3],[1,2],[3,2],[2,2],[1,1],[3,1],[2,1],[1,0],[3,0],[0,3],[4,3],[0,2],[4,2],[0,1],[4,1]]
  """

  meanR = np.array(x)[0,0:22,10]
  koord = [[2, 4], [0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [-1, 2], [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [0.9, 0], [2, 0], [3, 0], [2, -1]]
  x,y = [],[]
  for i in koord:
      x.append(i[0])
      y.append(i[1])

  z = meanR

  xi = numpy.linspace(-2, 6, N)
  yi = numpy.linspace(-2, 6, N)
  zi = scipy.interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

  # set points > radius to not-a-number. They will not be plotted.
  # the dr/2 makes the edges a bit smoother
  dr = xi[1] - xi[0]
  for i in range(N):
      for j in range(N):
          r = numpy.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
          if (r - dr/2) > radius:
              zi[j,i] = "nan"

  # make figure
  fig = plt.figure()

  # set aspect = 1 to make it a circle
  ax = fig.add_subplot(111, aspect = 1)

  # use different number of levels for the fill and the lines
  CS = ax.contourf(xi, yi, zi, 60, cmap = plt.cm.jet, zorder = 1)
  ax.contour(xi, yi, zi, 15, colors = "grey", zorder = 2)

  # make a color bar
  cbar = fig.colorbar(CS, ax=ax)

  # add the data points
  # I guess there are no data points outside the head...
  ax.scatter(x, y, marker = 'o', c = 'k', s = 15, zorder = 3)

  # draw a circle
  # change the linewidth to hide the 
  circle = matplotlib.patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "none")
  ax.add_patch(circle)

  # make the axis invisible 
  for loc, spine in ax.spines.items():
      # use ax.spines.items() in Python 3
      spine.set_linewidth(0)

  # remove the ticks
  ax.set_xticks([])
  ax.set_yticks([])

  # Add some body parts. Hide unwanted parts by setting the zorder low
  # add two ears
  circle = matplotlib.patches.Ellipse(xy = [0,2], width = 0.5, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
  ax.add_patch(circle)
  circle = matplotlib.patches.Ellipse(xy = [4,2], width = 0.5, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
  ax.add_patch(circle)
  # add a nose
  xy = [[1.5,3], [2,4.5],[2.5,3]]
  polygon = matplotlib.patches.Polygon(xy = xy, facecolor = "w", zorder = 0)
  ax.add_patch(polygon) 

  # set axes limits
  ax.set_xlim(-0.5, 4.5)
  ax.set_ylim(-0.5, 4.5)

  plt.show() 

if __name__=='__main__':
  test()
  
  #train_test()
  #batch_crop('C:/Users/Richard/Documents/MATLAB/fig4/')
  #test('C:/Users/Richard/Documents/MATLAB/fig3/')
  #image_matrix_processing('C:/Users/Richard/Documents/MATLAB/image matrix_grasp/')
"""
path = 'C:/Users/Richard/Documents/MATLAB/fig2/'

files = os.listdir(path)
dic = {}
for file in files:
  print(file)
  im_frame = Image.open(path+file)
  im_frame = im_frame.crop((200,150,600,500))
  im_frame.save('intra_croped/'+file, format='png')
  np_frame = np.array(im_frame.getdata())
  sf = file.split('_')
  if sf[0] not in dic:
    dic[sf[0]] = {'x': [], 'y': []}
  dic[sf[0]]['x'].append(np_frame)
  dic[sf[0]]['y'].append(int(sf[3]))

for k, v in dic.items():
  f = open('fig_dat_intra_crop/' + k + '_topo-polt.pic', 'wb')
  pickle.dump(v, f)
  f.close()

path = 'C:/Users/Richard/Documents/MATLAB/fig/'
def img_proc():
  im_frame = Image.open(path+'bwyu_twist_69_1_1.5s.png')
  area1 = (200,150,600,500)
  area2 = (150, 50, 650, 580)
  area3 = (250, 200, 550, 450)

  cropped_img_1 = im_frame.crop(area1)
  #cropped_img_2 = im_frame.crop(area2)
  #cropped_img_3 = im_frame.crop(area3)
  
  img1 = cropped_img_1.resize((32, 32))
  img2 = cropped_img_1.resize((64, 64))
  img3 = cropped_img_1.resize((128, 128))
  img4 = cropped_img_1.resize((256, 256))

  img1.save('ss1.png', format='png')
  img2.save('ss2.png', format='png')
  img3.save('ss3.png', format='png')
  img4.save('ss4.png', format='png')

  

img_proc()


def white_remover(x):
  for i in range(len(x)):
    rev_x = []
    for j in range(len(x[i])):
      if x[i][j][0] != 255 or x[i][j][1] != 255 or x[i][j][2] != 255:
        rev_x.append(x[i][j])
  return rev_x

def lab_maker(y):
  result = []
  for i in range(len(y)):
    if y[i] is 1:
      result.append([1, 0, 0])
    elif y[i] is 2:
      result.append([0, 1, 0])
    else:
      result.append([0, 0, 1])
  return result







f = open('fig_dat_v2/bwyu_topo-polt_v2.pic', 'rb')
dat = pickle.load(f)
f.close()

x = np.array(dat['x'])
y = np.array(lab_maker(dat['y']))

x, y = white_remover(x)

kv = BCI.gen_kvt_idx(y, 10)

acc = []
for train_idx, val_idx, test_idx in kv:
  x_train, x_test = x[train_idx], x[test_idx]
  x_val, y_val = x[val_idx], y[val_idx]
  y_train, y_test = y[train_idx], y[test_idx]
  model = RCNN.create_model()
  x_train = x_train.reshape(x_train.shape[0], 574, 1000, 3)
  x_val = x_val.reshape((x_val.shape[0], 574, 1000, 3))
  x_test = x_test.reshape((x_test.shape[0], 574, 1000, 3))

  model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1000)

  model.evaluate(x_test, y_test)

"""