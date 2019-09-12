#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2018 Design, Analysis and Research Corporation - All Rights Reserved. INTERNAL USE ONLY. Permission
required for distribution from Dr. Willem Anemaat, President DARcorporation.

This file contains utility functions.
"""
import numpy as np
import random
import string

from typing import Optional, Union, Iterable

array_like = Union[float, Iterable[float], np.ndarray]

rng = random.SystemRandom()
digs = string.digits + string.ascii_letters


def str2float(s: str) -> Union[float, type(np.nan)]:
    try:
        return float(s.strip())
    except ValueError:
        return np.nan


def int2base(x: int, base: int) -> str:
    """Convert a decimal integer into any base, represented as a string.

    Parameters
    ----------
    x : int
        Integer to be converted.
    base : int
        Base to convert into.

    Returns
    -------
    str
        String representation of the integer in the given base.
    """
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append("-")

    digits.reverse()

    return "".join(digits)


def get_random_key(length: int = 4, leading_char: Optional[str] = None) -> str:
    """Generate a random alphanumeric string with a given number of characters.

    Parameters
    ----------
    length : int
        Length of the key. Default is 4.
    leading_char : str, optional
        Leading character.

    Returns
    -------
    str
        Random alphanumeric string.
    """
    res = ("{:0>" + str(length) + "s}").format(
        int2base(rng.randint(0, 36 ** length - 1), 36)
    )
    if leading_char is not None:
        return "".join([leading_char, res])
    return res


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


def first_nonzero(arr: np.ndarray, axis: int) -> Union[np.ndarray, Iterable, tuple]:
    """Find the index of the first non-zero value in an array along a given axis.

    Parameters
    ----------
    arr : np.ndarray
        Numpy array.
    axis : int
        Axis along which to search.

    Returns
    -------
    ind : int, optional
        Index of the first non-zero value, or None if no non-zero value exists.
    """
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), None)


def last_nonzero(arr: np.ndarray, axis: int) -> Union[np.ndarray, Iterable, tuple]:
    """Find the index of the last non-zero value in an array along a given axis.

    Parameters
    ----------
    arr : np.ndarray
        Numpy array.
    axis : int
        Axis along which to search.

    Returns
    -------
    ind : int, optional
        Index of the last non-zero value, or None if no non-zero value exists.
    """
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, None)
