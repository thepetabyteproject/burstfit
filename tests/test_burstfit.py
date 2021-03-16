import os

import numpy as np

from burstfit.data import BurstData
from burstfit.fit import BurstFit
from burstfit.model import Model, SgramModel
from burstfit.utils.functions import pulse_fn, gauss_norm2, sgram_fn

_install_dir = os.path.abspath(os.path.dirname(__file__))
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pytest


@pytest.fixture(scope="module", autouse=True)
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


@pytest.fixture(scope="module", autouse=True)
def bf(bd):
    pm = Model(pulse_fn)
    sm = Model(gauss_norm2)
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


@pytest.fixture(scope="module", autouse=True)
def bf_fitted(bf):
    bf.fitall(spectra_bounds=([50, 5, 200, 5, 0], [150, 50, 300, 50, 1]), plot=False)
    return bf


def test_validate(bf):
    bf.validate()


def test_precalc(bf, bd):
    bf.precalc()
    assert bf.nf == 336
    assert bf.nt == 156
    assert bf.profile_param_names == ["S", "mu", "sigma", "tau"]
    assert bf.spectra_param_names == ["mu1", "sig1", "mu2", "sig2", "amp1"]
    assert bf.metadata[2] == bd.dm
    assert pytest.approx(np.argmax(bf.ts), rel=1) == 79


def test_profile_fit(bf):
    bf.precalc()
    bf.initial_profilefit(plot=False)
    assert list(bf.profile_params.keys()) == [1]
    assert pytest.approx(bf.profile_params[1]["popt"][0], abs=0.1) == 512.8
    assert bf.profile_params[1]["popt"][3] < 1
    assert bf.profile_params[1]["popt"][2] < 1


def test_make_spectra_w_profile_params(bf):
    bf.precalc()
    bf.initial_profilefit(plot=False)
    bf.make_spectra()
    assert pytest.approx(np.trapz(bf.spectra), abs=0.1) == 1


def test_make_spectra_wo_profile_params(bf):
    bf.precalc()
    bf.make_spectra()
    assert pytest.approx(np.trapz(bf.spectra), abs=0.1) == 1


def test_initial_spectra_fit(bf):
    bf.precalc()
    bf.initial_profilefit(plot=False)
    bf.make_spectra()
    bf.initial_spectrafit(
        bounds=([50, 5, 200, 5, 0], [150, 50, 300, 50, 1]), plot=False
    )
    assert list(bf.spectra_params.keys()) == [1]
    assert pytest.approx(bf.spectra_params[1]["popt"][0], abs=1) == 87
    assert pytest.approx(bf.spectra_params[1]["popt"][2], abs=1) == 284


def test_fitall(bf):
    bf.fitall(spectra_bounds=([50, 5, 200, 5, 0], [150, 50, 300, 50, 1]), plot=False)
    assert bf.ncomponents == 1
    assert bf.sgram_params[1]["popt"] == bf.sgram_params["all"][1]["popt"]
    assert pytest.approx(bf.sgram_params[1]["popt"][0], rel=1) == 74
    assert pytest.approx(bf.sgram_params[1]["popt"][2], rel=1) == 281
    assert pytest.approx(bf.sgram_params[1]["popt"][5], rel=10) == 560
    assert pytest.approx(bf.sgram_params[1]["popt"][-1], rel=1) == 474


def test_red_chisq(bf_fitted):
    red_chi_sq = bf_fitted.calc_redchisq()
    assert pytest.approx(red_chi_sq, rel=0.1) == 1


def test_model(bf_fitted):
    m = bf_fitted.model
    assert m.shape == (336, 156)
    assert 79 == pytest.approx(np.argmax(m.sum(0)), rel=1)


def test_model_from_params(bf_fitted):
    m = bf_fitted.model_from_params([0], *bf_fitted.sgram_params[1]["popt"])
    m = m.reshape((bf_fitted.nf, bf_fitted.nt))
    assert 79 == pytest.approx(np.argmax(m.sum(0)), rel=1)


def test_run_mcmc(bf_fitted):
    mcmc_kwargs = {
        "nwalkers": 20,
        "nsteps": 100,
        "skip": 500,
        "ncores": 4,
        "start_pos_dev": 0.01,
        "prior_range": 0.8,
        "save_results": True,
        "outname": "test_h5",
    }
    bf_fitted.run_mcmc(plot=True, **mcmc_kwargs)
    real_vals = [73.97, 28.77, 285.61, 52.4, 0.38, 584.88, 78.93, 0.75, 0.15, 474.45]
    real_errs = [3.5, 4.5, 9, 11, 0.05, 60, 0.17, 0.082, 0.11, 0.22]
    for i in range(len(real_errs)):
        assert (
            pytest.approx(bf_fitted.mcmc_params[1]["popt"][i], abs=2 * real_errs[i])
            == real_vals[i]
        )

    assert os.path.isfile("test_h5_samples.h5")
    os.remove("test_h5_samples.h5")
