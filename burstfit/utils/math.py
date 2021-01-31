import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def f_test(x, y):
    """
     F-Test
    Args:
        x: Input array 1
        y: Input array 2

    Returns:

    """
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1) / np.var(y, ddof=1)  # calculate F test statistic
    dfn = x.size - 1  # define degrees of freedom numerator
    dfd = y.size - 1  # define degrees of freedom denominator
    p = 1 - stats.f.cdf(f, dfn, dfd)  # find p-value of F test statistic
    return f, p


def tests(off_pulse, on_pulse_res, pth=0.05, ntest=1):
    """
    Run statistical tests to compare the two inputs

    Args:
        off_pulse: input array to compare
        on_pulse_res: input array to compare
        pth: threshold on p value to consider the distributions similar
        ntest: minimum number of tests to pass

    Returns:

    """
    off_pulse = off_pulse.ravel()
    on_pulse_res = on_pulse_res.ravel()
    pv_ttest = stats.ttest_ind(on_pulse_res, off_pulse, equal_var=False).pvalue
    pv_kruskal = stats.kruskal(on_pulse_res, off_pulse).pvalue
    pv_ks = stats.kstest(on_pulse_res, off_pulse).pvalue
    pv_ftest = f_test(on_pulse_res, off_pulse)[1]
    logging.info(
        f"P values: T-test ({pv_ttest:.5f}), Kruskal ({pv_kruskal:.5f}), "
        f"KS ({pv_ks:.5f}), F-test ({pv_ftest:.5f})"
    )

    m = np.array([pv_ttest, pv_kruskal, pv_ks, pv_ftest]) > pth
    if m.sum() >= ntest:
        return 1
    else:
        return 0


def fma(param, m, a):
    """

    Args:
        param: parameter value
        m: number to multiply to param
        a: number to add

    Returns:

    """
    return param * m + a
