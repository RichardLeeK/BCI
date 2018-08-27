
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

def y_generator(y):
  new_y = []
  for v in y:
    if v == 'tongue': new_y.append(0)
    elif v == 'foot': new_y.append(1)
    elif v == 'right': new_y.append(2)
    elif v == 'left': new_y.append(3)
  return new_y


if __name__ == '__main__':
  path = 'data/'
  files = os.listdir(path)
  for file in files:
    x, y = load_one_data(path + file)

