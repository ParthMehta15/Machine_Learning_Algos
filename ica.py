'''
Independent Component Analysis

Use Case:
Can be used to seperate mixed signals (audio) into seperate signals.
Load the signls and stack them row-wise.
Eg. load 4 wav files using librosa, stack them and pass the matrix through the function.
  The output will also have 4 rows. Each row is a seperated signal (audio).
  Can be saved as audio file using librosa.


'''


import numpy as np

def ica(data_matrix):   #data_matrix (size: MxN)

  means = np.mean(data_matrix, axis =1)
  means = np.expand_dims(means,1)

  M = np.subtract(data_matrix, means)
  R = np.matmul(M,M.T)

  e, s, e_t = np.linalg.svd(R)
  diag_s = np.diag(s)

  lambda_s = np.linalg.inv(np.power(diag_s,0.5))

  C = lambda_s@e.T
  X = np.matmul(C,M)

  norm_vec = []
  for i in range(X.shape[1]):
    norm_vec.append((np.linalg.norm(X[:,i]))**2)
  norm_vec = np.array(norm_vec)

  D = (norm_vec*X)@X.T
  b, s2, b_t = np.linalg.svd(D)

  A = b.T@C
  H = A@M

  return H