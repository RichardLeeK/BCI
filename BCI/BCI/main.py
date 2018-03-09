import BCI
import os

os.chdir('C:/Users/Cnmlab/Documents/MATLAB/dat/S2/twist/')
files = os.listdir()

for file in files:
  if file.split('.')[-1] == 'csv':
    BCI.one_routine('C:/Users/Cnmlab/Documents/MATLAB/dat/S2/twist/', file, 'twist')