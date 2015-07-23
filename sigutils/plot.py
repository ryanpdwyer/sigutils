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

from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from sigutils._util import (freqresp, freqz, lin_or_logspace, mag_phase)



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


def find_crossings(x, a=0):
    """Return array indices where x - a changes sign.

    See http://stackoverflow.com/a/29674950/2823213"""
    x = np.atleast_1d(x)
    return np.where(np.diff(np.signbit(x - a)))[0]


def find_repeated_roots(x):
    """"""
    cnt = Counter()
    x_iterable = list(x)
    while x_iterable != []:
        xi = x_iterable[0]
        compared_equal = np.isclose(xi, x_iterable)
        equal_indices = np.nonzero(compared_equal)[0]
        for i in equal_indices[::-1]:
            x_iterable.pop(i)

        cnt[xi] = np.sum(compared_equal)

    return {key: val for key, val in cnt.items() if val > 1}


def _x_per_inch(ax):
    """Conversion factor between the plot x variable and the figure width.

    For example, """
    xlim = ax.get_xlim()
    return (xlim[1] - xlim[0]) / ax.get_figure().get_figwidth()


def _y_per_inch(ax):
    ylim = ax.get_ylim()
    return (ylim[1] - ylim[0]) / ax.get_figure().get_figheight()


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

    rcParamsDefault =   {'figure.figsize' : (8,6),
                         'lines.linewidth': 1.5,
                         'figure.dpi'     : 300,
                         'savefig.dpi'    : 300,
                         'axes.labelsize'      : 12,}

    if rcParams is not None:
        rcParamsDefault.update(rcParams)

    with mpl.rc_context(rcParamsDefault):
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
            gain_index = find_crossings(mag, gain_point)
            for i in gain_index:
                ax1.axvline(x=freq[i], color='k',  linestyle='--')
                ax2.axvline(x=freq[i], color='k',  linestyle='--')

        ax1.set_ylabel('Magnitude [dB]')
        ax2.set_ylabel('Phase [deg.]')
        ax2.set_xlabel('Frequency')
        fig.tight_layout()
        return fig, (ax1, ax2)


def bodes(freq, resp,  xlim=None, xlog=True, mag_lim=None, phase_lim=None,
          gain_point=None, figax=None, rcParams=None):
    for f, r in zip(freq, resp):
        figax = bode(f, r, xlim=xlim, xlog=xlog, mag_lim=mag_lim,
                     phase_lim=phase_lim, gain_point=gain_point,
                     figax=figax, rcParams=rcParams)

    return figax


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
        If given, draws a vertical line on the bode plot at this point
    figax : tuple of (fig, (ax1, ax2)), optional
        The figure and axes to create the plot on, if given. If omitted, a new
        figure and axes are created
    rcParams : dictionary, optional
        matplotlib rc settings dictionary"""
    freq, resp = freqresp(system, xlim=xlim, N=N, xlog=xlog)

    return bode(freq, resp, xlim=xlim, xlog=xlog, mag_lim=mag_lim,
                phase_lim=phase_lim, gain_point=gain_point,
                figax=figax, rcParams=rcParams)


def bode_syss(systems, xlim=None, N=10000, xlog=True, mag_lim=None,
              phase_lim=None, gain_point=None, figax=None, rcParams=None):
    """Make a nice bode plot for the given system.

    Parameters
    ----------
    systems : an iterable containing instances of the LTI class or a tuple
    describing the system. The following gives the number of elements
    in the tuple and the interpretation:

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
        If given, draws a vertical line on the bode plot at this point
    figax : tuple of (fig, (ax1, ax2)), optional
        The figure and axes to create the plot on, if given. If omitted, a new
        figure and axes are created
    rcParams : dictionary, optional
        matplotlib rc settings dictionary"""
    for i, system in enumerate(systems):
        # Create the figure if none is given
        if figax is None and i == 0:
            figax = bode_sys(system, xlim=xlim, N=N, xlog=xlog, mag_lim=mag_lim,
                             phase_lim=phase_lim, gain_point=gain_point,
                             figax=figax, rcParams=rcParams)
        else:
            bode_sys(system, xlim=xlim, N=N, xlog=xlog, mag_lim=mag_lim,
                     phase_lim=phase_lim, gain_point=gain_point,
                     figax=figax, rcParams=rcParams)

    return figax


def bode_z(b, a=1, fs=1, xlim=None, N=1000, xlog=False, mag_lim=None,
           phase_lim=None, gain_point=None, figax=None, rcParams=None):
    """Make a nice bode plot for a discrete time system.

    Parameters
    ----------"""
    freq, resp = freqz(b=b, a=a, fs=fs, xlim=xlim, N=N, xlog=xlog)

    return bode(freq, resp, xlim=xlim, xlog=xlog, mag_lim=mag_lim,
                phase_lim=phase_lim, gain_point=gain_point,
                figax=figax, rcParams=rcParams)


def bode_firs(bs, fs=1, xlim=None, N=1000, xlog=False, mag_lim=None,
              phase_lim=None, gain_point=None, figax=None, rcParams=None):
    for i, b in enumerate(bs):
        if figax is None and i == 0:
            figax = bode_z(b, a=1, fs=fs, xlim=xlim, N=N, xlog=xlog,
                           mag_lim=mag_lim, phase_lim=phase_lim,
                           gain_point=gain_point, figax=figax,
                           rcParams=rcParams)
        else:
            bode_z(b, a=1, fs=fs, xlim=xlim, N=N, xlog=xlog,
                   mag_lim=mag_lim, phase_lim=phase_lim,
                   gain_point=gain_point, figax=figax,
                   rcParams=rcParams)

    return figax


def bode_zz(systems, fs=1, xlim=None, N=1000, xlog=False, mag_lim=None,
            phase_lim=None, gain_point=None, figax=None, rcParams=None):
    """"""
    for i, system in enumerate(systems):
        b = system[0]
        if len(system) == 1:
            a = 1
        elif len(system) == 2:
            a = system[1]
        else:
            raise ValueError(
                "Digital system ({0}) has more than two elements.".format(
                    system))

        if figax is None and i == 0:
            figax = bode_z(b, a, fs=fs, xlim=xlim, N=N, xlog=xlog,
                           mag_lim=mag_lim, phase_lim=phase_lim,
                           gain_point=gain_point, figax=figax,
                           rcParams=rcParams)
        else:
            bode_z(b, a, fs=fs, xlim=xlim, N=N, xlog=xlog,
                   mag_lim=mag_lim, phase_lim=phase_lim,
                   gain_point=gain_point, figax=figax,
                   rcParams=rcParams)

    return figax


def bode_an_dig(analogs, digitals, fs, xlim=None, N=1000, xlog=False,
               mag_lim=None, phase_lim=None, gain_point=None, figax=None,
               rcParams=None):
    """Plots analog and digital systems together on the same axes."""
    figax = bode_syss(analogs, N=N, xlim=xlim, xlog=xlog, mag_lim=mag_lim,
                      phase_lim=phase_lim, gain_point=gain_point,
                      figax=figax, rcParams=rcParams)
    bode_zz(digitals, fs=fs, xlim=xlim, xlog=xlog, mag_lim=mag_lim,
                      phase_lim=phase_lim, gain_point=gain_point,
                      figax=figax, rcParams=rcParams)
    return figax


def _font_pt_to_inches(x):
    """Convert points to inches (1 inch = 72 points)."""
    return x / 72.


def _pole_zero(z, p, k, xlim=None, ylim=None, figax=None, rcParams=None):
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)

    rcParamsDefault =   {'figure.figsize' : (6,6),
                         'lines.linewidth': 1.5,
                         'figure.dpi'     : 300,
                         'savefig.dpi'    : 300,
                         'axes.labelsize'      : 12,}
    if rcParams is not None:
        rcParamsDefault.update(rcParams)

    with mpl.rc_context(rcParamsDefault):
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax

        markersize = mpl.rcParams['lines.markersize']
        markeredgewidth = mpl.rcParams['lines.markeredgewidth']

        zeros, = ax.plot(z.real, z.imag, linewidth=0, marker='o',
                         markerfacecolor=None,)
        poles, = ax.plot(p.real, p.imag, linewidth=0, color=zeros.get_color(),
                         marker ='x', markeredgewidth=3.5*markeredgewidth,
                         markersize=markersize*1.5)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        circ = plt.Circle((0, 0), radius=1, linewidth=1,
                          fill=False, color='gray')
        ax.axvline(0, color='k')
        ax.axhline(0, color='k')
        ax.add_patch(circ)
        ax.grid()

        x_per_inch = _x_per_inch(ax)
        y_per_inch = _y_per_inch(ax)

        m_f = mpl.rcParams['font.size']
        m_z = zeros.get_markersize()

        m_inch_z = _font_pt_to_inches(m_z/2. + m_f/2.)

        m_x_z = m_inch_z * x_per_inch
        m_y_z = m_inch_z * y_per_inch

        m_p = poles.get_markersize()
        m_inch_p = _font_pt_to_inches(m_p/2. + m_f/2.)
        m_x_p = m_inch_p * x_per_inch
        m_y_p = m_inch_z * y_per_inch

        rep_zeros = find_repeated_roots(z)
        rep_poles = find_repeated_roots(p)

        for pt, val in rep_zeros.items():
            ax.text(pt.real + m_x_z, pt.imag + m_y_z, str(val))

        for pt, val in rep_poles.items():
            ax.text(pt.real + m_x_p, pt.imag + m_y_p, str(val))

        return fig, ax


def pole_zero(sys, xlim=None, ylim=None, figax=None, rcParams=None):
    if len(sys) == 2:
        z, p, k = signal.tf2zpk(*sys)
    elif len(sys) == 3:
        z, p, k = sys
    elif len(sys) == 4:
        z, p, k = signal.ss2zpk(*sys)
    else:
        ValueError("""\
sys must have 2 (transfer function), 3 (zeros, poles, gain),
or 4 (state space) elements. sys is: {}""".format(sys))

    return _pole_zero(z, p, k, xlim=xlim, ylim=ylim, figax=figax,
                      rcParams=rcParams)


def nyquist(freq, resp, freq_lim=None, xlim=None, ylim=None,
            figax=None, rcParams=None):
    if rcParams is None:
        rcParams = {'figure.figsize': (6, 6),
                    'lines.linewidth': 1.5,
                    'figure.dpi': 300,
                    'savefig.dpi': 300,
                    'font.size': 12}

    with mpl.rc_context(rcParams):
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax

        if freq_lim is not None:
            resp = resp[np.logical_and(freq > freq_lim[0], freq < freq_lim[1])]
        else:
            resp = resp

        ax.plot(resp.real, resp.imag, '-')
        circ = plt.Circle((0, 0), radius=1, linewidth=1, fill=False,
                          color='gray')
        ax.axvline(0, color='0.5')
        ax.axhline(0, color='0.5')
        ax.add_patch(circ)
        ax.grid()

        if xlim is not None:
            xlim = list(ax.get_xlim())
            if xlim[0] > -1.1:
                xlim[0] = -1.1

            ax.set_xlim(xlim)
        else:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

    return fig, ax
