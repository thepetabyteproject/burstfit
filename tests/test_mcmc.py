import pytest
from burstfit.mcmc import MCMC
from burstfit.data import BurstData
from burstfit.fit import BurstFit
from burstfit.model import Model, SgramModel
from burstfit.utils.functions import pulse_fn_vec, gauss_norm2, sgram_fn_vec
import numpy as np
import os

_install_dir = os.path.abspath(os.path.dirname(__file__))

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
    pnames = ["S", "mu_t", "sigma_t", "tau"]
    snames = ["mu_f1", "sigma_f1", "mu_f2", "sigma_f2", "amp"]
    pm = Model(pulse_fn_vec, param_names=pnames)
    sm = Model(gauss_norm2, param_names=snames)
    sgmodel = SgramModel(pm, sm, sgram_fn_vec)
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
    bf.precalc()
    return bf


@pytest.fixture(scope="module", autouse=True)
def mcmc(bf, bd):
    params = [
        74.6600995552445,
        29.08997449280042,
        282.0987558671014,
        48.04757054335587,
        0.4123476364901036,
        555.8960256802608,
        79.02660306937426,
        0.7844582964946152,
        0.07026178702689054,
        474.5754998470141,
    ]
    param_names = []
    for i in range(1, 2):
        for p in bf.param_names:
            param_names += [p + "_" + str(i)]
    mcmc = MCMC(
        bf.model_from_params,
        bf.sgram,
        params,
        param_names,
        nwalkers=20,
        nsteps=100,
        skip=200,
        start_pos_dev=0.01,
        prior_range=0.9,
        ncores=4,
        outname="test",
        save_results=True,
    )
    return mcmc


def test_init(mcmc):
    assert mcmc.nwalkers == 20
    assert mcmc.nsteps == 100
    assert mcmc.pos.shape == (mcmc.nwalkers, 10)
    assert mcmc.ndim == 10
    pos = mcmc.pos
    ig = mcmc.initial_guess
    assert (np.abs((pos.min(axis=0) - ig) / ig) < mcmc.start_pos_dev).sum() == 10
    assert (np.abs((pos.max(axis=0) - ig) / ig) < mcmc.start_pos_dev).sum() == 10


def test_lnprior(mcmc):
    assert mcmc.lnprior(mcmc.initial_guess) == 0
    assert np.isinf(mcmc.lnprior(mcmc.min_prior - 0.1 * mcmc.initial_guess))
    assert np.isinf(mcmc.lnprior(mcmc.max_prior + 0.1 * mcmc.initial_guess))


def test_lnprob_lnlk(mcmc):
    assert np.isinf(mcmc.lnprob(mcmc.min_prior - 0.2 * mcmc.initial_guess))

    p = mcmc.lnprob(mcmc.min_prior + 0.1 * mcmc.initial_guess)
    assert pytest.approx(p, rel=0.1) == -26215.37

    lk = mcmc.lnprob(mcmc.min_prior + 0.1 * mcmc.initial_guess)
    assert pytest.approx(lk, rel=0.1) == -26215.37


def test_priors(mcmc):
    mcmc.set_priors()

    # sigma_t and tau
    assert mcmc.min_prior[-2] == 0
    assert mcmc.min_prior[-3] == 0
    assert mcmc.max_prior[-2] == mcmc.max_prior[-3]

    # S
    assert mcmc.min_prior[-5] == 0
    assert mcmc.max_prior[-5] > np.max(mcmc.sgram.sum(0)) * mcmc.max_prior[-3]

    # sigma f
    assert mcmc.min_prior[1] == 0
    assert mcmc.min_prior[3] == 0
