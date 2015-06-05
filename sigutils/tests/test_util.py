# -*- coding: utf-8 -*-
import numpy as np
from sigutils._util import log_bins

def test_log_bins():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    x_out_ex = np.array([1, 2.5, 5.5])
    y_out_ex = np.array([1, 1, 1])
    return log_bins(x, y, r=2)