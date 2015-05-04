# -*- coding: utf-8 -*-
"""
============================
sigutils
============================
"""

from sigutils.plot import bode, bode_sys, bode_z, bode_syss

# Versioneer versioning
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
