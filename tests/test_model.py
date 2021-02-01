import numpy as np
import pytest

from burstfit.model import *
from burstfit.utils.functions import pulse_fn, spectra_fn, sgram_fn, gauss


@pytest.fixture(scope="function", autouse=True)
def model():
    model = Model(gauss)
    return model


def test_model_class(model):
    # model = Model(gauss)
    assert len(model.param_names) == 3
    assert model.nparams == 3


def test_model_evaluate(model):
    # model = Model(gauss)
    x = np.linspace(0, 100, 100)
    params = [10, 50, 5]
    y = model.evaluate(x, *params)
    assert pytest.approx(np.trapz(y), abs=0.1) == 10


def test_model_dict(model):
    d = model.get_param_dict(*[10, 50, 5])
    assert list(d.keys()) == ["S", "mu", "sigma"]
    assert d["S"] == 10
    assert d["mu"] == 50
    assert d["sigma"] == 5


def test_sgram_model():
    sgrammodel = SgramModel(
        pulse_model=pulse_fn, spectra_model=spectra_fn, sgram_fn=sgram_fn
    )
    assert sgrammodel.nparams == 7
