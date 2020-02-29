import numpy as np
from numpy import linalg as LA


# Q1.1
pd_Matrix = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

print("My PD Matrix is: \n", pd_Matrix, "\n")


# Q1.2
pd_sigma, pd_eigenvector = LA.eig(pd_Matrix)

print("The eigenvalue of the matrix is: ", round(pd_sigma[0], 2), ", ", round(pd_sigma[1], 2), ", ",
      round(pd_sigma[2], 2), "\n")
print("The correspoding eigenvoector is: ", np.around(pd_eigenvector[0], 2), ", ", np.around(pd_eigenvector[1], 2),
      ", ", np.around(pd_eigenvector[2], 2), "\n")


# Q1.3
# Since we see that all the eigenvalue of the pd_Matrix is positive, (3.41, 2, 0.59 > 0), then the matrix is indeed
# positive definite.


# Problem 2
A = pd_Matrix

# 2.1
L = LA.cholesky(A)
print("The Cholesky Decomposition of matrix A is L = \n", np.around(L, 2) , "\n")
print("Lu = L*v*A*v^(-1)= \n", np.matmul(np.matmul(np.matmul(L, pd_eigenvector), A), np.transpose(pd_eigenvector)),
      "\n")


# 2.2
s1 = np.random.normal(0, 1, 100000)
s2 = np.random.normal(0, 1, 100000)
s3 = np.random.normal(0, 1, 100000)
u = np.array([s1, s2, s3])

cov_u = np.cov(u)

print("Covariance matrix of 100,000 sampled data from normal distribution is close to Identity matrix: ",
      np.allclose(np.around(cov_u, 2), np.identity(3), rtol=1e-2), "\n")
# Since the diagonal of the covariance matrix is the variance of the sampled data, ie \sigma_{nn} = var(x_{n}), and all the data are sampled from standard normal distribution, that is, all the variance of x_{nn} = 1, and the diagonal of the covariance matrix is 1. Besides, since \sigma_{nm} = cov(x_{n},x_{m}), and all three X are sampled independently, so when m \neq n, cov(x_{n},x_{m}) \to 0, that is, all the matrix besides the diagonal approaches 0 when the sample size goes large enough. In conclusion, the covariance matrix is approximately the identity matrix.


# 2.3
# ?

# 3.1

