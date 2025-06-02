# -*- coding: utf-8 -*-
"""Compatibility wrapper for the Phase 4 helper functions.

This module was previously located at the repository root as
``phase4_functions.py``. It now lives in the :mod:`phase4` package. Importing
from this file ensures backward compatibility with existing scripts and
documentation.

"""

from phase4.functions import *  # noqa: F401,F403
