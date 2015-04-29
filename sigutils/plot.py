# -*- coding: utf-8 -*-
"""
"""
from __future__ import division, print_function, absolute_import

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def mag_phase(z, dB=True, degrees=True):
    mag = np.abs(z)
    phase = np.unwrap(np.angle(z))
    if dB:
        mag = 20 * np.log10(mag)
    if degrees:
        phase = phase * 180 / np.pi

    return mag, phase

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
def bode(freq, rep, xlim=None, mag_lim=None, phase_lim=None, gain_point=None, figax=None, xlog=True):
    """Make a nice bode plot for the given frequency, magnitude, and phase data.

    Parameters
    ----------
    freq : array
        Array of frequencies used for the Bode plot
    rep : array
        Complex response evaluated at the frequencies in freq
    gain_point : float, optional
        If given, draws a vertical line on the bode plot at 
    xlim : tuple of (x_min, x_max), optional
        Minimum and maximum values (x_min, x_max) of the plot's x-axis
    mag_lim : tuple of (mag_min, mag_max, mag_delta), optional
        A three element tuple containing the magnitude axis minimum, maximum
        and tick spacing
    phase_lim : tuple of (phase_min, phase_max, phase_delta), optional
        A three element tuple containing the phase axis minimum, maximum
        and tick spacing
    figax : tuple of (fig, (ax1, ax2)), optional
        The figure and axes to create the plot on, if given. If omitted, a new
        figure and axes are created
    xlog : bool, optional
        Use a log (True) or linear (False) scale for the x-axis


    Returns
    -------
    figax : tuple of (fig, (ax1, ax2))
        The figure and axes of the bode plot

    """
    mag, phase = mag_phase(rep, dB=True, degrees=True)
    with mpl.rc_context({'figure.figsize' : (8,6),
                         'lines.linewidth': 1.5,
                         'figure.dpi'     : 300,
                         'savefig.dpi'    : 300,
                         'font.size'      : 16,}):
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