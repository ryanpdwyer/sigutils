"""
Frequency Domain Least Squares
==============================

This algorithm tries to approximate an analytic transfer function :math:`H(s)`.
See digital signal processing book.

- Pick an analytic transfer function H(s)
- Select the numerator order N and denominator order D
- Define M separate input u_m coside sequences, each of length N + 1
- Compute M output y_m cosine sequences, each of length D 
- X = ( y(-1)...y(-D) u(0)...u(-N) )
- Y = A_m cos(phi_m)
- Compute the psuedo-inverse 

"""

import numpy as np

def butter_lp(f, f0):
    return 1/(1+f*1j/f0)

# Let's approximate this with a 1st order top and bottom filter function
# def fdls(N, D, M):
#     k = np.arange(-N, 0.5)

    # np.arange()

# A few lines on the frequency domain least squares algorithm
# See http://dx.doi.org/10.1109/MSP.2007.273077
# import numpy.linalg
# fs = 1000
# f0 = 10
# m = 8192
# n = 513
# d = 0
# f = np.linspace(-0.5, 0.5, m) // All frequencies
# tm = np.arange(-n,0.5,1)     // All times
# zf = butter_lp(f, f0/fs)
# af = np.abs(zf)
# pf = -1 * np.angle(zf)
# np.cos(2*np.pi*f[0]*tm)
# f2d, t2d = np.meshgrid(f, t)
# u = np.cos(2*np.pi*f2d*t2d)
# X = u
# Y = af*np.cos(pf)
# X1 = np.linalg.pinv(X)
# out = np.dot(Y, X1)