import Competition as com
import CSP as csp
import numpy as np

def data_load(file, filtering=True):
  path = 'E:/Richard/EEG/Competition/raw/'
  cnt, mrk, dur, y = com.load_cnt_mrk_y(path + file)
  if filtering:
    cnt = csp.arr_bandpass_filter(cnt, 8, 30, 250, 5)
  epo = com.cnt_to_epo(cnt, mrk, dur)
  epo, y = com.out_label_remover(epo, y)
  return epo, y

def normalization(epo):
  ne = []
  for e in epo:
    mean = np.mean(e)
    std = np.std(e)
    ne.append((e - mean) / std)
  return np.array(ne)
    
def sax_conversion(nx):
  from saxpy.sax import sax_via_window
  for i in range(nx.shape[0]):
    for j in range(nx.shape[2]):
      sax = sax_via_window(nx[0][:][2], 5, 5, 3, 'none', 0.01)
  return sax

def ax_conversion(nx):
  import matplotlib.pyplot as plt
  for i in range(nx.shape[0]):
    for j in range(nx.shape[2]):
      for k in range(0, 25):
        x = np.linspace(0, 75, 75)
        A = np.vstack([x, np.ones(75)]).T
        m, c = np.linalg.lstsq(A, nx[0,k*75:(k+1)*75,2])[0]
        plt.plot(x, m*x+c, 'r-', x, nx[0,k*75:(k+1)*75,2], 'b*')
        plt.show()

        



        
if __name__ == '__main__':
  x, y = data_load('A01T.npz', False)
  nx = normalization(x)
  sax = ax_conversion(nx)
  print('abc')
