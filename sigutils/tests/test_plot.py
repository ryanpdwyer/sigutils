# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import unittest

import numpy as np
from numpy.testing import assert_allclose
from sigutils.plot import (mag_phase, bode, lin_or_logspace)


def test_mag_phase():
    z = np.array([1+0j, (1-1j)/np.sqrt(2), -1j])
    exp_phase = np.array([0, -45, -90])
    exp_mag = np.array([0, 0, 0])
    mag, phase = mag_phase(z, dB=True, degrees=True)
    assert_allclose(mag, exp_mag, atol=1e-12)
    assert_allclose(phase, exp_phase, atol=1e-12)

def test_lin_or_logspace():
    x_min = 1
    x_max = 10
    n = 10
    exp = np.arange(1, 11)
    assert_allclose(exp, lin_or_logspace(x_min, x_max, n, False))


class Test_bode(unittest.TestCase):
    def setUp(self):
        self.freq = np.logspace(0, 4, 51)
        self.resp = 1/(1 + 1j * self.freq / 100)

    def test_bode(self):
        bode(self.freq, self.resp)
