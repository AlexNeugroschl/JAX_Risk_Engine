"""
A comparison shared by the Greeks tests (tests/test_greeks.py, tests/test_greeks_bermudan.py,
tests/test_sensitivity_ore_parity.py), scaled to the compared magnitude.
"""
import numpy as np


def assert_close(actual, expected, rtol, err_msg=""):
    """`actual` equals `expected` to `rtol` of each entry and of `expected`'s largest entry."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * float(np.max(np.abs(expected))),
                               err_msg=err_msg)
