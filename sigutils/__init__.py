# -*- coding: utf-8 -*-
"""
============================
sigutils
============================
"""
import numpy as np
from scipy import signal

from sigutils.plot import (bode, bodes, bode_sys, bode_syss,
                           bode_z, bode_firs, bode_zz,
                           bode_an_dig, nyquist,
                           magtime_z, magtime_zz, magtime_firs, pole_zero)

from sigutils._util import log_bins, lin_bins, freqresp, freqz



def fft(x, t=1, real=True, window='rect'):
    """Compute the FFT of x using window. Return the FFT, and the
    FFT frequencies.

    x : array
        The input data to Fourier transform
    t : float or array, optional
        Time information, used to adjust the Fourier transform frequencies.
        If t is a number, use it as the time step. If t is an array,
        infer the timestep from the array.
    real : bool, optional
        Compute only non-negative frequency coefficients, which contain all
        the information if x is real.
    window : str or array
        Window to use; defaults to 'rect' (no window).
    """
    if isinstance(t, int) or isinstance(t, float):
        dt = t
    else:
        dt = t[1] - t[0]

    if real:
        fft_func = np.fft.rfft
        fft_freq_func = np.fft.rfftfreq
    else:
        fft_func = np.fft.fft
        fft_freq_func = np.fft.fftfreq

    window_factor = signal.get_window(window, x.size)

    ft = fft_func(x * window_factor)
    ft_freq = fft_freq_func(x.size, d=dt)

    return ft, ft_freq

# Versioneer versioning
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

