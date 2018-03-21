
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

mu, sigma = 100, 15
tmep = [1, 2, 3, 1, 2, 3, 2, 3, 2, 5, 7, 4, 0, 0, 0, 8, 9, 1, 2, 3, 4, 10]

n, bins, patches = plt.hist(tmep, 5)
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y)
plt.show()