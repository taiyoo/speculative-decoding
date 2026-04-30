"""Compatibility visualization entrypoint.

All plotting/evaluation backend logic now lives in ``visual_eval_backend.py``.
This module re-exports the public API so existing imports from ``visualize``
continue to work.
"""

from visual_eval_backend import *
