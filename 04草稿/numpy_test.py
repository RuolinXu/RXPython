import numpy as np

a = np.ones(100)
b= np.concatenate((a, np.ones(100)*1.5),axis=0)
print(a)
print(b)