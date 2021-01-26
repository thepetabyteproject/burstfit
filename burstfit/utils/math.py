import scipy
import numpy as np
import logging

logger = logging.getLogger(__name__)


def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1) / np.var(y, ddof=1)  # calculate F test statistic
    dfn = x.size - 1  # define degrees of freedom numerator
    dfd = y.size - 1  # define degrees of freedom denominator
    p = 1 - scipy.stats.f.cdf(f, dfn, dfd)  # find p-value of F test statistic
    return f, p


def tests(off_pulse, on_pulse_res, pth=0.05):
    off_pulse = off_pulse.ravel()
    on_pulse_res = on_pulse_res.ravel()
    pv_ttest = scipy.stats.ttest_ind(on_pulse_res, off_pulse, equal_var=False).pvalue
    pv_kruskal = scipy.stats.kruskal(on_pulse_res, off_pulse).pvalue
    pv_ks = scipy.stats.kstest(on_pulse_res, off_pulse).pvalue
    pv_ftest = f_test(on_pulse_res, off_pulse)[1]
    logging.info(
        f"P values: T-test ({pv_ttest:.5f}), Kruskal ({pv_kruskal:.5f}), "
        f"KS ({pv_ks:.5f}), F-test ({pv_ftest:.5f})"
    )

    m = np.array([pv_ttest, pv_kruskal, pv_ks, pv_ftest]) > pth
    if m.sum() > 0:
        return 1
    else:
        return 0


def fma(param, m, a):
    return param * m + a
