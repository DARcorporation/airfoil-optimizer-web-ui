#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains utility functions.
"""
import numpy as np

from typing import Optional, Union, Iterable

array_like = Union[float, Iterable[float], np.ndarray]


def str2float(s: str) -> Union[float, type(np.nan)]:
    try:
        return float(s.strip())
    except ValueError:
        return np.nan


def cosspace(
    start: Optional[float] = None, end: Optional[float] = None, n: Optional[int] = None
) -> np.ndarray:
    """Creates a cosine-spaced array of `n` points between `start` and `end`.

    Parameters
    ----------
    start, end : float, optional
        Start and end values. If only `start` is given, values are varied from 0 to this value. If neither `start`
        nor `end` is given, values are varied from 0 to 1.
    n : int, optional
        Number of points. If unspecified, a hundred points will be generated.

    Returns
    -------
    np.ndarray
        Array of cosine spaced points between `start` and `end`.
    """
    if start is None:
        start = 0
        if end is None:
            end = 1
    elif end is None:
        end = start
        start = 0

    if n is None:
        n = 100

    r = (end - start) / 2.0
    theta = np.linspace(0, np.pi, n)

    return r + start - r * np.cos(theta)
