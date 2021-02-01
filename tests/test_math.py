import numpy as np

from burstfit.utils.math import f_test, fma
from burstfit.utils.math import tests as te


def test_ftest():
    a = np.random.normal(loc=0, scale=1, size=1000)
    b = np.random.normal(loc=0, scale=1, size=1000)
    f, p = f_test(a, b)
    assert p > 0.1

    c = np.random.uniform(size=1000)
    f, p = f_test(a, c)
    assert p < 1e-3


def test_stat_tests():
    a = np.random.normal(loc=0, scale=1, size=1000)
    b = np.random.normal(loc=0, scale=1, size=1000)
    res = te(a, b, ntest=2)
    assert res == 1

    c = np.random.uniform(size=1000)
    res = te(a, c, ntest=2)
    assert res == 0


def test_fma():
    r = fma(2, 2, 1)
    assert r == 5
