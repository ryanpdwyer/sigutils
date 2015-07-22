# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
import numpy as np
import pandas as pd


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


def lin_bins(x, y, n=None, dx=None):

    df = pd.DataFrame({'x': x, 'y': y})
    if n is None and dx is None:
        raise ValueError("Must specify n or dx")
    elif n is not None and dx is not None:
        raise ValueError("Specify only one of n or dx")
    elif n is not None:
        df['bins'] = df.index // n
    elif dx is not None:
        df['bins'] = (df.x - df.x.min()) // dx

    xy = df.groupby('bins').mean()

    return xy.x.values, xy.y.values
