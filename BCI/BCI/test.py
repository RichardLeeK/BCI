import scipy.stats as ss
import numpy as np

def read_data(path):
  file = open(path)
  lines = file.readlines()
  file.close()
  dic = {}
  for line in lines[1:]:
    sl = line.split(',')
    pid = int(sl[0].split('_')[0])
    data_len = float(sl[1])
    dic[pid] = data_len

  return dic

def main():
  no_data = read_data('tmp/No.csv')
  abp_data = read_data('tmp/oABP.csv')
  icp_data = read_data('tmp/oICP.csv')
  all_data = read_data('tmp/ALL.csv')

  pen = open('tmp/data_length.csv', 'w')
  pen.write('FileName,ABP artifact,ICP artifact,AorI\n')
  for k, v in no_data.items():
    if k not in abp_data:
      continue
    if k not in icp_data:
      continue
    if k not in all_data:
      continue

    no_val = no_data[k]
    abp_val = abp_data[k]
    icp_val = icp_data[k]
    all_val = all_data[k]

    abp_artifact = no_val - abp_val
    icp_artifact = no_val - icp_val
    all_artifact = no_val - all_val

    pen.write(str(k) + ',' + str(abp_artifact) + ',' + str(icp_artifact) + ',' + str(all_artifact) + '\n')

  pen.close()


def MUT(x, y):
  #Mann-Whitney U test
  (U, p) = ss.mannwhitneyu(x, y)
  return p

def PTT(x, y):
  t = ss.ttest_rel(x, y)
  return t.pvalue

def spss():
  file = open('tmp/SPSS.csv')
  lines = file.readlines()
  RCNN = []
  FD = []
  LDA = []
  LP = []
  RLDAS = []
  for line in lines[1:]:
    sl = line.split(',')
    RCNN.append(float(sl[1]))
    FD.append(float(sl[2]))
    LDA.append(float(sl[3]))
    LP.append(float(sl[4]))
    RLDAS.append(float(sl[5]))
  pen = open('SPSS_result.csv', 'w')
  pen.write('Test,FD,LDA,LP,RLDAS\n')
  pen.write('MUT,'+str(MUT(RCNN, FD))+','+str(MUT(RCNN, LDA))+','+str(MUT(RCNN, LP))+','+str(MUT(RCNN,RLDAS))+'\n')
  pen.write('PPT,'+str(PTT(RCNN, FD))+','+str(PTT(RCNN, LDA))+','+str(PTT(RCNN, LP))+','+str(PTT(RCNN, RLDAS))+'\n')
  pen.close()

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


def mibif(x, y):
  fb_csp = CSP.filterbank_CSP(x)
  xs = [[], [], [], []]
  for i in range(len(fb_csp)):
    xs[y.argmax(axis=1)[i]].append(fb_csp[i])
  mis = np.zeros((len(xs), len(xs[0][0]), len(xs[0]) + 10))
  for i in range(len(xs)): # class number
    for j in range(len(xs[0][0])): # filter number
      for k in range(len(xs[i])): # epoch count
        one = xs[i][k][j]
        rest = []
        for l in range(len(xs)):
          if i == l: continue
          try:
            rest.extend(xs[l][k][j])
          except Exception as e:
            print(e)
        mis[i][j][k] = mutual_information(one, rest)
  return np.sum(np.sum(mis, axis=2), axis=0)

def make_plot():
  import numpy as np
  import matplotlib.pyplot as plt

  bar1 = [0.706298669,0.512972582,0.810844987,0.810844987]
  bar2 = [0.735260041,0.522692844,0.834244166,0.834244166]
  bar3 = [0.70510582,0.510224868,0.810469577,0.810469577]
  bar4 = [0.698510998,0.481737271,0.805419026,0.805419026]
  
  yer1 = [0.046212297,0.072766472,0.023064386,0.023064386]
  yer2 = [0.042870652,0.078064932,0.022132357,0.022132357]
  yer3 = [0.046591546,0.073620136,0.02443552,0.02443552]
  yer4 = [0.0472298,0.076059316,0.023763944,0.023763944]

  bandwirth = 0.3
  r1 = np.arange(len(bar1)) * 2
  r2 = [x + bandwirth for x in r1]
  r3 = [x + bandwirth for x in r2]
  r4 = [x + bandwirth for x in r3]

  plt.bar(r1, bar1, width=bandwirth, yerr=yer1, capsize=3)
  plt.bar(r2, bar2, width=bandwirth, yerr=yer2, capsize=3)
  plt.bar(r3, bar3, width=bandwirth, yerr=yer3, capsize=3)
  plt.bar(r4, bar4, width=bandwirth, yerr=yer4, capsize=2)

  plt.savefig('fig/res.eps', format='eps', dpi=1000)


from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass_filter(data, lowcut, highcut, fs, order=5):
  from scipy.signal import butter, lfilter
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  y = lfilter(b, a, data)
  return y

def arr_bandpass_filter(data, lowcut, highcut, fs, order=5):
  y = np.array(data)
  for i in range(len(data)):
    for j in range(len(data[i])):
      cur_data = data[i][j]
      cur_y = bandpass_filter(cur_data, lowcut, highcut, fs, order)
      y[i][j] = cur_y
  return y


def real_test():
  file = open('dlfkjalk.csv', 'r')
  lines = file.readlines()
  x = []; y = []
  for line in lines:
    sl = line.split(",")
    cur_x = []
    for v in sl[:-3]:
      cur_x.append(float(v))
    cur_y = [float(sl[-3]), float(sl[-2]), float(sl[-1])]
    x.append(np.array(cur_x))
    y.append(np.array(cur_y))

  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
  x = np.array(x)
  y = np.array(y).argmax(axis=1)
  clf2.fit(x[20:], y[20:])
  sc  = clf2.score(x[:20], y[:20])
  print(sc)


if __name__ == "__main__":
    real_test()
    make_plot()

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0
    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')


    plt.figure(3)
    plt.plot(t, x, label='Noisy signal')
    y_ = bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y_, label='dfdfd')
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')




    plt.show()