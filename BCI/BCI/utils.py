from sklearn.metrics import cohen_kappa_score

def get_cohen_kappa(y1, y2):
  
  return cohen_kappa_score(y1, y2, labels=None, weights=None)

