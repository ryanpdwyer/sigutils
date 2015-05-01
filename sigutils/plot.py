# -*- coding: utf-8 -*-
"""

- We should have plotting functions

    bode_ba(ba, ...)
        Takes an analog transfer function in ba form
    bode_z(b, a=1, fs, ...)
        Takes a digital transfer function in z form. Is fs, nyq, or dt preferred?
    bode_zpk(zpk, fs?, ...)
        Use zpk form (or state space?)
    bode_s(sympy_expression, var, ...)
        Takes a sympy expression, var, evaulates it at 2*pi*f...
    bode_f(func, ...)
        Takes a function, which will be evaluated to determine the response



"""
from __future__ import division, print_function, absolute_import

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

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


def adjust_y_ticks(ax, delta):
    """Adjust the y-axis tick marks on ax to the spacing delta."""
    ylim = np.array(ax.get_ylim()) / delta
    ymin = ylim[0]//1  # Round towards - infinity
    ymax = -(-ylim[1]//1)  # Trick to round towards + infinity
    # Note: this rounds away from zero so we never make the axis limits smaller
    ax_new_lim = np.array([ymin, ymax]) * delta
    ax_new_ticks = np.arange(ax_new_lim[0], ax_new_lim[1]+1, delta)
    ax.set_ybound(*ax_new_lim)
    ax.set_yticks(ax_new_ticks)



# To do: should db=True be an option?
def bode(freq, resp, xlim=None, xlog=True, mag_lim=None, phase_lim=None,
         gain_point=None, figax=None, rcParams=None):
    """Make a nice bode plot for the given frequency, magnitude, and phase data.

    Parameters
    ----------
    freq : array
        Array of frequencies used for the Bode plot
    resp : array
        Complex response evaluated at the frequencies in freq
    xlim : tuple of (x_min, x_max), optional
        Minimum and maximum values (x_min, x_max) of the plot's x-axis
    xlog : bool, optional
        Use a log (True) or linear (False) scale for the x-axis
    mag_lim : tuple of (mag_min, mag_max, mag_delta), optional
        A three element tuple containing the magnitude axis minimum, maximum
        and tick spacing
    phase_lim : tuple of (phase_min, phase_max, phase_delta), optional
        A three element tuple containing the phase axis minimum, maximum
        and tick spacing
    gain_point : float, optional
        If given, draws a vertical line on the bode plot at 
    figax : tuple of (fig, (ax1, ax2)), optional
        The figure and axes to create the plot on, if given. If omitted, a new
        figure and axes are created
    rcParams : dictionary, optional
        matplotlib rc settings dictionary

    Returns
    -------
    figax : tuple of (fig, (ax1, ax2))
        The figure and axes of the bode plot

    """
    mag, phase = mag_phase(resp, dB=True, degrees=True)
    if rcParams is None:
        rcParams = {'figure.figsize' : (8,6),
                         'lines.linewidth': 1.5,
                         'figure.dpi'     : 300,
                         'savefig.dpi'    : 300,
                         'font.size'      : 16,}
    with mpl.rc_context(rcParams):
        if figax is None:
            fig, (ax1, ax2) = plt.subplots(nrows=2)
        else:
            fig, (ax1, ax2) = figax

        # Light grey major y gridlines
        ax1.yaxis.grid(True, linestyle='-', color='.8')
        ax2.yaxis.grid(True, linestyle='-', color='.8')

        if xlog:
            ax1.semilogx(freq, mag)
            ax2.semilogx(freq, phase)
        else:
            ax1.plot(freq, mag)
            ax2.plot(freq, phase)

        if xlim is not None:
            ax1.set_xlim(*xlim)
            ax2.set_xlim(*xlim)

        if mag_lim is not None:
            ax1.set_ylim(mag_lim[0], mag_lim[1])
            adjust_y_ticks(ax1, mag_lim[2])

        if phase_lim is not None:
            ax2.set_ylim(phase_lim[0], phase_lim[1])
            adjust_y_ticks(ax2, phase_lim[2])
    
        if gain_point is not None:
            gain_index = find_gain(mag, gain_point)
            ax1.axvline(x=freq[gain_index], color='k',  linestyle='--')
            ax2.axvline(x=freq[gain_index], color='k',  linestyle='--')

        
        ax1.set_ylabel('Magnitude [dB]')
        ax2.set_ylabel('Phase [deg.]')
        ax2.set_xlabel('Frequency')
        fig.tight_layout()
        return fig, (ax1, ax2)


def bode_sys(system, xlim=None, N=10000, xlog=True, mag_lim=None,
             phase_lim=None, gain_point=None, figax=None, rcParams=None):
    """Make a nice bode plot for the given system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
    The following gives the number of elements in the tuple and
    the interpretation:

        * 2 (num, den)
        * 3 (zeros, poles, gain)
        * 4 (A, B, C, D)
    xlim : tuple of (x_min, x_max), optional
        Minimum and maximum values (x_min, x_max) of the plot's x-axis
    N : int, optional
        The number of points to calculate the system response at
    xlog : bool, optional
        Use a log (True) or linear (False) scale for the x-axis
    mag_lim : tuple of (mag_min, mag_max, mag_delta), optional
        A three element tuple containing the magnitude axis minimum, maximum
        and tick spacing
    phase_lim : tuple of (phase_min, phase_max, phase_delta), optional
        A three element tuple containing the phase axis minimum, maximum
        and tick spacing
    gain_point : float, optional
        If given, draws a vertical line on the bode plot at 
    figax : tuple of (fig, (ax1, ax2)), optional
        The figure and axes to create the plot on, if given. If omitted, a new
        figure and axes are created
    rcParams : dictionary, optional
        matplotlib rc settings dictionary"""
    if xlim is None:
        w, resp = signal.freqresp(system, n=N)
        # Recalculate the data points over a linear interval if requested
        if not xlog:
            w = np.linspace(w.min(), w.max(), N)
            _ , resp = signal.freqresp(system, w)
    else:
        w = 2 * np.pi * lin_or_logspace(xlim[0], xlim[1], N, xlog)
        _ , resp = signal.freqresp(system, w)

    freq = w / (2 * np.pi)
    return bode(freq, resp, xlim=xlim, xlog=xlog, mag_lim=mag_lim,
                phase_lim=phase_lim, gain_point=gain_point,
                figax=figax, rcParams=rcParams)


def bode_z(b, a=1, fs=1, xlim=None, N=1000, xlog=False, mag_lim=None,
           phase_lim=None, gain_point=None, figax=None, rcParams=None):
    if xlim is None:
        w, resp = signal.freqz(b, a, N)
    else:
        w = np.pi * lin_or_logspace(xlim[0], xlim[1], N, xlog) / fs
        _, resp = signal.freqz(b, a, worN=w)

    freq = w * fs / np.pi
    return bode(freq, resp, xlim=xlim, xlog=xlog, mag_lim=mag_lim,
                phase_lim=phase_lim, gain_point=gain_point,
                figax=figax, rcParams=rcParams)
