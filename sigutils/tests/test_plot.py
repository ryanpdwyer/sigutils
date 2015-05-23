# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import unittest

import numpy as np
from numpy.testing import assert_allclose
from sigutils.plot import (mag_phase, bode, lin_or_logspace, find_crossings,
                           find_repeated_roots)


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


def test_find_crossings():
    to_test = (
                ([0, 1, 3, 4], [1, -1, 1, 0.1, -0.1, 1, 10]),
              )

    for exp, x in to_test:
        assert_allclose(exp, find_crossings(x))


def test_find_repeated_roots():
    x = np.array([1-1j, 1+1j, 1-1j])
    out = {(1-1j): 2}
    assert find_repeated_roots(x) == out


# def test_x_per_inch():
#     assert False


# def test_y_per_inch():
#     assert False


class Test_bode_related_plots(unittest.TestCase):
    """These plotting classes are hard to test.
       Let's start with just calling each of the plotting functions with a
       typical signature to verify that there are no weird typos or errors."""
    def setUp(self):
        self.freq = np.logspace(0, 4, 51)
        self.resp = 1/(1 + 1j * self.freq / 100)

    def test_bode(self):
        bode(self.freq, self.resp)
