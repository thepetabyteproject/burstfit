import numpy as np

from burstfit.utils.math import f_test, fmae, transform_parameters
from burstfit.utils.math import tests as te


def test_ftest():
    a = np.random.normal(loc=0, scale=1, size=1000)
    f, p = f_test(a, a)
    assert p > 0.1

    c = np.random.uniform(size=1000)
    f, p = f_test(a, c)
    assert p < 1e-3


def test_stat_tests():
    a = np.random.normal(loc=0, scale=1, size=1000)
    res = te(a, a, ntest=2)
    assert res == 1

    b = np.random.normal(loc=0, scale=1, size=1000)
    res = te(a, b, ntest=1)
    assert res == 1

    res = te(a, b, ntest=2)
    assert res == 1

    c = np.random.uniform(size=1000)
    res = te(a, c, ntest=2)
    assert res == 0


def test_fmae():
    r, e = fmae(2, 2, 1, 0, 0, 0)
    assert r == 5
    assert e == 0


def test_transform_params():
    params = {}
    params["popt"] = [10, 20, 30]
    params["perr"] = [1, 5, 6]
    param_names = ["mu_f", "sigma_f", "S"]

    mapping = {
        "F_0": ["mu_f", 1.5, 0, 1400, 0],
        "F_sig": ["sigma_f", 1.5, 0, 0, 0],
        "Fluence": ["S", 2.5, 0.5, 0, 0],
    }
    physical_dict = transform_parameters(params, mapping, param_names)

    assert (
        physical_dict["popt"]["F_0"]
        == params["popt"][0] * mapping["F_0"][1] + mapping["F_0"][3]
    )
    assert (
        physical_dict["popt"]["F_sig"]
        == params["popt"][1] * mapping["F_sig"][1] + mapping["F_sig"][3]
    )
    assert (
        physical_dict["popt"]["Fluence"]
        == params["popt"][2] * mapping["Fluence"][1] + mapping["Fluence"][3]
    )

    assert physical_dict["perr"]["F_0"] == params["perr"][0] * mapping["F_0"][1]
    assert physical_dict["perr"]["F_sig"] == params["perr"][1] * mapping["F_sig"][1]

    p = 30 * 2.5
    p_err = p * np.sqrt((6 / 30) ** 2 + (0.5 / 2.5) ** 2)

    assert physical_dict["perr"]["Fluence"] == p_err
