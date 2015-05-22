# -*- coding: utf-8 -*-
"""
============================
sigutils
============================
"""

from sigutils.plot import (bode, bode_sys, bode_syss,
                           bode_z, bode_firs, bode_zz,
                           bode_an_dig, nyquist, pole_zero)

# Versioneer versioning
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
