# -*- coding: utf-8 -*-
"""
============================
sigutils
============================
"""

from sigutils.plot import (bode, bodes, bode_sys, bode_syss,
                           bode_z, bode_firs, bode_zz,
                           bode_an_dig, nyquist,
                           magtime_z, magtime_zz, magtime_firs, pole_zero)

from sigutils._util import log_bins, lin_bins, freqresp, freqz

# Versioneer versioning
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
