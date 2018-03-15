import numpy as np
from scipy.stats import kurtosis
# Raw signal analysis

def get_aac(x_list):
  sum = 0
  for i in range(1, len(x_list)):
    sum += (x_list[i] - x_list[i - 1])
  return sum / len(x_list)

def get_dasdv(x_list):
  sum = 0
  for i in range(1, len(x_list)):
    sum += (x_list[i] - x_list[i - 1]) * (x_list[i] - x_list[i - 1])
  return np.sqrt(sum / len(x_list))

def get_iav(x_list):
  sum = 0
  for i in range(len(x_list)):
    sum += x_list[i]
  return sum

def get_logd(x_list):
  sum = 0
  for i in range(len(x_list)):
    np.log(np.abs(x_list[i]))
  return np.e ** (sum / len(x_list))

def get_mav(x_list):
  sum = 0
  for i in range(len(x_list)):
    sum += np.abs(x_list[i])
  return sum / len(x_list)

def get_mlogk(x_list):
  sum = 0
  for i in ragne(len(x_list)):
    sum += x_list[i]
  return np.abs(sum) / len(x_list)

def get_rms(x_list):
  sum = 0
  for i in range(len(x_list)):
    sum += (x_list[i] * x_list[i])
  return np.sqrt(sum / len(x_list))

def get_kurt(x_list):
  return kurtosis(x_list)

def get_ssc(x_list):
  ## ...
  sum = 0
  for i in range(2, len(x_list)):
    if (x_list[i] - x_list[i - 1]) > 0.1:
      sum += 1
    else:
      sum -= 1
  return sum

def get_ssi(x_list):
  sum = 0
  for i in range(len(x_list)):
    sum += x_list[i] * x_list[i]
  return sum

def get_var(x_list):
  sum = 0
  for i in range(len(x_list)):
    sum += x_list[i] * x_list[i]
  return sum / (len(x_list) - 1)

def get_wfL(x_list):
  sum = 0
  for i in range(1, len(x_list)):
    sum += (x_list[i] - x_list[i - 1])
  return sum

def get_zcs(x_list):
  ## 
  return 0

def get_tms(x_list, order=3):
  sum = 0
  for i in range(len(x_list)):
    sum += (x_list[i] ** order)
  return np.abs(sum) / len(x_list)

def get_ar4(x_list):
  ##
  return 0

def get_std(x_list):
  sum = 0
  for i in range(len(x_list)):
    sum += x_list[i]
  return np.sqrt((sum / (len(x_list) - 1)))

def get_mval(x_list):
  return np.average(x_list)

def get_map(x_list):
  return np.max(x_list)

def get_time_domain_features(x_list):
  result_dic = {}
  result_dic['AAC'] = get_aac(x_list)
  result_dic['DASDV'] = get_dasdv(x_list)
  result_dic['IAV'] = get_iav(x_list)
  result_dic['LOGD'] = get_logd(x_list)
  result_dic['MAV'] = get_mav(x_list)
  result_dic['mLOGK'] = get_mlogk(x_list)
  result_dic['RMS'] = get_rms(x_list)
  result_dic['KURT'] = get_kurt(x_list)
  #result_dic['SSC'] = get_ssc(x_list)
  result_dic['SSCI'] = get_ssi(x_list)
  result_dic['VAR'] = get_var(x_list)
  result_dic['WFL'] = get_wfl(x_list)
  #result_dic['ZCS'] = get_zcs(x_list)
  result_dic['TM3'] = get_tms(x_list, 3)
  result_dic['TM4'] = get_tms(x_list, 4)
  result_dic['TM5'] = get_tms(x_list, 5)
  #result_dic['AR4'] = get_ar4(x_list)
  result_dic['STD'] = get_std(x_list)
  result_dic['MVAL'] = get_mval(x_list)
  result_dic['MAP'] = get_map(x_list)
  return result_dic

