# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from sigutils import fft


class TestFFT(unittest.TestCase):
    def test_fft_exact_bin(self):
        x = np.array([0, 1, 0, -1, 0, 1, 0, -1])
        ft, ft_freq = fft(x, t=1, real=True, window='rect')
        ft_expected = np.array([0, 0, -4j, 0, 0])
        ft_freq_expected = np.arange(5) / 8
        assert_array_almost_equal(ft, ft_expected)
        assert_array_almost_equal(ft_freq, ft_freq_expected)
