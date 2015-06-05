# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
import numpy as np

def log_bins(x, y, r=1.5):
    """Average data over logrithmically spaced intervals of fractional size r. nice for plotting
    data on log-log plots. r controlls """
    if r <= 1:
        raise ValueError('r ({}) must be greater than 1.'.format(r))
    
    i = 0
    x_out = []
    y_out = []
    while x[i] == 0:
        i += 1
    
    while i < x.size:
        x0 = x[i] # Create mask from x0 to 2 x0
        mask = np.logical_and(x >= x0, x < r*x0)
        x_out.append(x[mask].mean())
        y_out.append(y[mask].mean())
        i = np.max(mask.nonzero()) + 1  # Reset the index
    
    x_out = np.array(x_out)
    y_out = np.array(y_out)
    return x_out, y_out
