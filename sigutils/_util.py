# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
import numpy as np
import pandas as pd


def log_bins(x, y, r=1.5):
    """Average data over logrithmically spaced intervals of fractional size r.
    Nice for plotting data on log-log plots."""
    if r <= 1:
        raise ValueError('r ({}) must be greater than 1.'.format(r))

    df = pd.DataFrame({'x': x, 'y': y})

    log_mask = df.x > 0

    x0 = df[log_mask].x.min()
    df['bins'] = (np.log(df.x) - np.log(x0)) // np.log(r)

    xy = df[log_mask].groupby('bins').mean()

    return xy.x.values, xy.y.values


def lin_bins(x, y, dx=None, n=None):
    """Average data over linearly spaced intervals, specified in
    x units (dx) or numper of points to average together (n).

    Helpful for averaging noisy data for clearer plotting.
    """

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
