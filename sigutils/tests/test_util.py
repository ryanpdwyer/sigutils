# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_almost_equal
from sigutils._util import log_bins, lin_bins


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
