import xgboost as xgb
import Competition as com
import CSP as csp
import numpy as np
import pandas as pd
import BCI as bci

def test2():
  from sklearn.datasets import load_boston
  boston = load_boston()
  print(boston.keys())
  data = pd.DataFrame(boston.data)
  data.columns = boston.feature_names
  X, y = data.iloc[:,:-1],data.iloc[:,-1]
  data_dmatrix = xgb.DMatrix(data=X,label=y)
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
  xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
  xg_reg.fit(X_train,y_train)

  preds = xg_reg.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, preds))
  print("RMSE: %f" % (rmse))
  print('abc')



def test(filtering=True):
  import os, scipy
  from sklearn.metrics import accuracy_score
  os.chdir('E:/Richard/EEG/Competition/4c_f/')
  i = 1
  for i in range(1, 10):
    print(i)
    c = scipy.io.loadmat('psd/A0'+str(i)+'.mat')['psd'][0][0]
    for j in range(4):
      ctx = np.transpose(c[0][j])
      cty = np.transpose(c[1][j]).argmax(axis=1)
      cvx = np.transpose(c[2][j])
      cvy = np.transpose(c[3][j]).argmax(axis=1)
      model = xgb.XGBClassifier()
      model.fit(ctx, cty)
      pred = model.predict(cvx)
      pen = open('psd.csv', 'a')
      pen.write(str(i) + ',' + str(j) + ',' + str(accuracy_score(cvy, pred)) + '\n')
      pen.close()




  



if __name__ == '__main__':

  test()