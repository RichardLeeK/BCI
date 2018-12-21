
from scipy.io import loadmat
import numpy as np
import os


class MotorImageryDataset:
    def __init__(self, dataset='A01T.npz'):
        if not dataset.endswith('.npz'):
            dataset += '.npz'

        self.data = np.load(dataset)

        self.Fs = 250 # 250Hz from original paper

        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

        self.raw = self.data['s'].T
        self.events_type = self.data['etyp'].T
        self.events_position = self.data['epos'].T
        self.events_duration = self.data['edur'].T
        self.artifacts = self.data['artifacts'].T

        # Types of motor imagery
        self.mi_types = {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue', 783: 'unknown'}

    def get_trials_from_channel(self, channel=7):

        # Channel default is C3

        startrial_code = 768
        starttrial_events = self.events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        trials = []
        classes = []
        for index in idxs:
            try:
                type_e = self.events_type[0, index+1]
                class_e = self.mi_types[type_e]
                classes.append(class_e)

                start = self.events_position[0, index]
                stop = start + self.events_duration[0, index]
                trial = self.raw[:, start:stop]
                # if you want to use one channel, try followings
                # trial = self.raw[channel, start:stop]
                trials.append(trial)

            except Exception as e:
                continue

        return trials, classes


def load_one_data(path):
  datasetA1 = MotorImageryDataset(dataset=path)
  trials, classes = datasetA1.get_trials_from_channel()
  return trials, y_generator(classes)

def load_cnt_mrk_y(path):
  ds = MotorImageryDataset(dataset=path)
  cnt = np.transpose(ds.raw)
  mrk = ds.events_position.flatten()
  dur = ds.events_duration.flatten()
  y = np.zeros(len(mrk))
  for i in range(len(mrk)):
    ds.events_type[:,i][0]
    if ds.events_type[:,i][0] not in ds.mi_types:
      y[i] = 5
    elif ds.mi_types[ds.events_type[:,i][0]] == 'left':
      y[i] = 0
    elif ds.mi_types[ds.events_type[:,i][0]] == 'right':
      y[i] = 1
    elif ds.mi_types[ds.events_type[:,i][0]] == 'foot':
      y[i] = 2
    elif ds.mi_types[ds.events_type[:,i][0]] == 'tongue':
      y[i] = 3
    else:
      y[i] = 4
  return cnt, mrk[:-1], dur[:-1], y[1:]

def y_generator(y):
  new_y = []
  for v in y:
    if v == 'tongue': new_y.append(0)
    elif v == 'foot': new_y.append(1)
    elif v == 'right': new_y.append(2)
    elif v == 'left': new_y.append(3)
  return new_y

def cnt_to_epo(cnt, mrk, dur):
  epo = []
  for i in range(len(mrk)):
    epo.append(np.array(cnt[mrk[i] : mrk[i] + dur[i]]))
  epo = np.array(epo)
  return epo

def out_label_remover(x, y):
  new_x = []; new_y = [];
  for i in range(len(y)):
    if y[i] == 4 or y[i] == 5:
      None
    else:
      new_x.append(np.array(x[i]))
      new_y.append(int(y[i]))
  return np.array(new_x), np.array(new_y)


if __name__ == '__main__':
  path = 'data/'
  files = os.listdir(path)
  for file in files:
    x, y = load_one_data(path + file)

