# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import unittest

import numpy as np
from numpy.testing import assert_allclose
from sigutils.plot import mag_phase

class Test_mag_phase(unittest.TestCase):
    @staticmethod
    def test_mag_phase():
        z = np.array([1+0j, (1-1j)/np.sqrt(2), -1j])
        exp_phase = np.array([0, -45, -90])
        exp_mag = np.array([0, 0, 0])
        mag, phase = mag_phase(z, dB=True, degrees=True)
        assert_allclose(mag, exp_mag, atol=1e-12)
        assert_allclose(phase, exp_phase, atol=1e-12)

