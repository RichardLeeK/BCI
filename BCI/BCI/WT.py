# Wavelet transform

import pywt, Competition, Com_test
import matplotlib.pyplot as plt
import numpy as np


def wavelet_transform():
  return 0



def batch(sub='1'):
  path = 'data/A0' + sub + 'T.npz'
  cnt, mrk, dur, y = Competition.load_cnt_mrk_y(path)
  epo = Com_test.cnt_to_epo(cnt, mrk, dur)
  epo, y = Com_test.out_label_remover(epo, y)
  wd = pywt.dwt2(epo, 'bior1.3')
  LL, (LH, HL, HH) = wd
  fig = plt.figure(figsize=(12, 3))
  titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
  for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
  fig.tight_layout()
  plt.show()

if __name__ == '__main__':
  batch()