import os

import numpy as np

from burstfit.data import BurstData
from burstfit.fit import BurstFit
from burstfit.model import Model, SgramModel
from burstfit.utils.functions import pulse_fn, spectra_fn, sgram_fn

_install_dir = os.path.abspath(os.path.dirname(__file__))
import pytest


@pytest.fixture(scope="function", autouse=True)
def bd():
    fil_file = os.path.join(_install_dir, "data/28.fil")
    bd = BurstData(
        fp=fil_file,
        dm=475.28400,
        tcand=2.0288800,
        width=2,
        snr=16.8128,
        min_samp=256,
    )
    bd.prepare_data()
    return bd


@pytest.fixture(scope="function", autouse=True)
def bf(bd):
    pm = Model(pulse_fn)
    sm = Model(spectra_fn)
    sgmodel = SgramModel(pm, sm, sgram_fn)
    bf = BurstFit(
        sgram_model=sgmodel,
        sgram=bd.sgram,
        dm=bd.dm,
        width=bd.width,
        foff=bd.foff,
        fch1=bd.fch1,
        tsamp=bd.tsamp,
        clip_fac=bd.clip_fac,
    )
    return bf


def test_validate(bf):
    bf.validate()


def test_precalc(bf, bd):
    bf.precalc()
    assert bf.nf == 336
    assert bf.nt == 156
    assert bf.profile_param_names == ["S", "mu", "sigma", "tau"]
    assert bf.spectra_param_names == ["nu_0", "nu_sig"]
    assert bf.metadata[2] == bd.dm


def test_profile_fit(bf):
    bf.precalc()
    bf.initial_profilefit()
    assert list(bf.profile_params.keys()) == [1]
    assert pytest.approx(bf.profile_params[1]["popt"][0], abs=0.1) == 511.1


def test_make_spectra_w_profile_params(bf):
    bf.precalc()
    bf.initial_profilefit()
    bf.make_spectra()
    assert pytest.approx(np.trapz(bf.spectra), abs=0.1) == 1


def test_make_spectra_wo_profile_params(bf):
    bf.precalc()
    bf.make_spectra()
    assert pytest.approx(np.trapz(bf.spectra), abs=0.1) == 1


def test_initial_spectra_fit(bf):
    bf.precalc()
    bf.initial_profilefit()
    bf.make_spectra()
    bf.initial_spectrafit()
    assert list(bf.spectra_params.keys()) == [1]
    assert pytest.approx(bf.spectra_params[1]["popt"][0], abs=1) == 299


def test_fitall(bf):
    bf.fitall()
    assert bf.ncomponents == 1


def test_model(bf):
    bf.fitall()
    m = bf.model
    assert m.shape == (336, 156)
