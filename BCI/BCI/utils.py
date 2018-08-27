from sklearn.metrics import cohen_kappa_score
import numpy as np

def get_cohen_kappa(y1, y2):
  
  return cohen_kappa_score(y1, y2, labels=None, weights=None)

def shannon_entropy(A, mode="auto", verbose=False):
    A = np.asarray(A)
    # Determine distribution type
    if mode == "auto":
        condition = np.all(A.astype(float) == A.astype(int))
        if condition:
            mode = "discrete"
        else:
            mode = "continuous"
    if verbose:
        print(mode, file=sys.stderr)
    # Compute shannon entropy
    pA = A / A.sum()
    # Remove zeros
    pA = pA[np.nonzero(pA)[0]]
    if mode == "continuous":
        return -np.sum(pA*np.log2(A))  
    if mode == "discrete":
        return -np.sum(pA*np.log2(pA)) 


def get_mutual_information(x, y, mode='auto', normalized=False):
    """
    I(X, Y) = H(X) + H(Y) - H(X,Y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # Determine distribution type
    if mode == "auto":
        condition_1 = np.all(x.astype(float) == x.astype(int))
        condition_2 = np.all(y.astype(float) == y.astype(int))
        if all([condition_1, condition_2]):
            mode = "discrete"
        else:
            mode = "continuous"

    H_x = shannon_entropy(x, mode=mode)
    H_y = shannon_entropy(y, mode=mode)
    H_xy = shannon_entropy(np.concatenate([x,y]), mode=mode)

    # Mutual Information
    I_xy = H_x + H_y - H_xy
    if normalized:
        return I_xy/np.sqrt(H_x*H_y)
    else:
        return  I_xy