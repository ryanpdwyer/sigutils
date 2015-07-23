# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from sigutils._util import log_bins, lin_bins, mag_phase, lin_or_logspace, freqz


def test_log_bins():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    x_out_ex = np.array([1, 2.5, 5.5])
    y_out_ex = np.array([1, 1, 1])
    x_out, y_out = log_bins(x, y, r=2)
    assert_array_almost_equal(x_out, x_out_ex)
    assert_array_almost_equal(y_out, y_out_ex)


def test_lin_bins():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y = np.array([0, 1, 4, 9, 16, 25, 36, 49])
    x_out_ex = np.array([0.5, 2.5, 4.5, 6.5])
    y_out_ex = np.array([0.5, 6.5, 20.5, 42.5])
    x_out, y_out = lin_bins(x, y, n=2)
    assert_array_almost_equal(x_out, x_out_ex)
    assert_array_almost_equal(y_out, y_out_ex)


def test_mag_phase():
    z = np.array([1+0j, (1-1j)/np.sqrt(2), -1j])
    exp_phase = np.array([0, -45, -90])
    exp_mag = np.array([0, 0, 0])
    mag, phase = mag_phase(z, dB=True, degrees=True)
    assert_allclose(mag, exp_mag, atol=1e-12)
    assert_allclose(phase, exp_phase, atol=1e-12)


def test_lin_or_logspace():
               # exp,            (x_min, x_max, n, log)
    to_test = (
               (np.arange(1,11),     (1, 10,  10, False)),
               (10**np.arange(0, 4), (1, 1000, 4, True))
               )

    for exp, args in to_test:
        assert_allclose(exp, lin_or_logspace(*args))

def test_freqz():
    b = [1]
    freqz(b, N=10)
