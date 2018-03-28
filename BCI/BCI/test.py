import scipy.stats as ss

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

  
  
spss()