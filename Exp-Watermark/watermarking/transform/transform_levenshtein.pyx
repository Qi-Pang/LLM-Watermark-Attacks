import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def transform_levenshtein(float[:] x, float[:] y, float gamma=1.0):
    cdef int i, j
    cdef float cost,tmp

    cdef int n = len(x)
    cdef int m = len(y)

    cdef np.ndarray[np.float32_t, ndim=2] npA = np.zeros((n+1,m+1), dtype=np.float32)
    cdef float[:,:] A = npA
    for i in range(0,n+1):
        for j in range(0,m+1):
            if i == 0:
                A[i][j] = j * gamma
            elif j == 0:
                A[i][j] = i * gamma
            else:
                tmp = x[i-1]-y[j-1]
                cost = sqrt(tmp*tmp)
                A[i][j] = A[i-1][j]+gamma
                if A[i][j-1]+gamma < A[i][j]:
                    A[i][j] = A[i][j-1]+gamma
                if A[i-1][j-1]+cost < A[i][j]:
                    A[i][j] = A[i-1][j-1]+cost

    return A[n][m]
