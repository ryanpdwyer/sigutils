# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy import signal

def cont2discrete(sys, dt, method='bilinear'):
    discrete_sys = signal.cont2discrete(sys, dt, method=method)[:-1]
    if len(discrete_sys) == 2:
        discrete_sys = tuple(np.squeeze(b_or_a) for b_or_a in discrete_sys)
    return discrete_sys

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


def mag_phase(z, dB=True, degrees=True):
    mag = np.abs(z)
    phase = np.unwrap(np.angle(z))
    if dB:
        mag = 20 * np.log10(mag)
    if degrees:
        phase = phase * 180 / np.pi

    return mag, phase


def lin_or_logspace(x_min, x_max, N, log):
    """Return a log or linearly spaced interval"""
    if log:
        if x_min <= 0 or x_max <= 0:
            raise ValueError(
                "x_min ({0}) and x_max ({1}) must be greater than 0.".format(
                    x_min, x_max))
        else:
            return np.logspace(np.log10(x_min), np.log10(x_max), N)
    else:
        return np.linspace(x_min, x_max, N)


def freqresp(system, xlim=None, N=10000, xlog=True):
    """Frequency response, possibly over a log range,
    of a continuous time system."""
    if xlim is None:
        w, resp = signal.freqresp(system, n=N)
        # Recalculate the data points over a linear interval if requested
        if not xlog:
            w = np.linspace(w.min(), w.max(), N)
            _, resp = signal.freqresp(system, w)
    else:
        w = 2 * np.pi * lin_or_logspace(xlim[0], xlim[1], N, xlog)
        _, resp = signal.freqresp(system, w)

    freq = w / (2 * np.pi)
    return freq, resp


def freqz(b, a=1, fs=1, xlim=None, N=1000, xlog=False):
    """Calculate the frequency response of a discrete time system over the
    range xlim, over a log or linear interval.

    Parameters
    ----------

    b : array-like
        Numerator coefficients of discrete time system
    a : array-like, optional
        Denominator coefficients of discrete time system
    fs : float, optional
        Sampling frequency; use to scale the output frequency array
    xlim : tuple of (x_min, x_max), optional
        Calculate the response from x_min to x_max. If omitted, the entire
        digital frequency axis is used
    N : int, optional
        The number of points to calculate the system response at
    xlog : bool, optional
        Calculate the frequency response at a log (True) or linearly spaced
        set of points"""
    # Squeeze arrays to deal with cont2discrete array issues
    b = np.squeeze(b)
    a = np.squeeze(a)
    if xlim is None:
        w, resp = signal.freqz(b, a)
        w = lin_or_logspace(w[w > 0][0], w[-1], N, True)
        _, resp = signal.freqz(b, a, w)
    else:
        w = 2 * np.pi * lin_or_logspace(xlim[0], xlim[1], N, xlog) / fs
        _, resp = signal.freqz(b, a, worN=w)

    freq = w * fs / (2 * np.pi)
    return freq, resp
