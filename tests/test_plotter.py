import os

import numpy as np

from burstfit.data import BurstData
from burstfit.fit import BurstFit
from burstfit.model import Model, SgramModel
from burstfit.utils.functions import pulse_fn, gauss_norm2, sgram_fn
from burstfit.utils.plotter import (
    plot_1d_fit,
    plot_2d_fit,
    plot_fit_results,
    autocorr_plot,
)

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
    bf.validate()
    bf.precalc()
    return bf


def test_plot_1d_fit():
    xdata = np.arange(100)
    ydata = np.linspace(0, 100, 100)
    plot_1d_fit(
        xdata,
        ydata,
        pulse_fn,
        [100, 20, 10, 5],
        xlabel="time",
        ylabel="amp",
        title="test",
        param_names=[],
        show=False,
        save=True,
        outname="1d_fit_res",
    )
    assert os.path.isfile("1d_fit_res.png")
    os.remove("1d_fit_res.png")


def test_plot_2d_fit(bf):
    popt = [73.97, 28.77, 285.61, 52.4, 0.38, 584.88, 78.93, 0.75, 0.15, 474.45]
    plot_2d_fit(
        bf.sgram,
        bf.sgram_model.evaluate,
        popt,
        bf.tsamp,
        title="test",
        show=False,
        save=True,
        outname="2d_fit_res",
        outdir=None,
    )
    assert os.path.isfile("2d_fit_res.png")
    os.remove("2d_fit_res.png")


def test_plot_fit_results(bf):
    popt = [73.97, 28.77, 285.61, 52.4, 0.38, 584.88, 78.93, 0.75, 0.15, 474.45]
    plot_fit_results(
        bf.sgram,
        bf.sgram_model.evaluate,
        popt,
        bf.tsamp,
        bf.fch1,
        bf.foff,
        mask=None,
        outsize=None,
        title="test",
        show=False,
        save=True,
        outname="test",
        outdir=None,
        vmin=1,
        vmax=30,
    )
    assert os.path.isfile("test_fit_results.png")
    os.remove("test_fit_results.png")


def test_autocorr_plot():
    n = np.linspace(0, 10000, 100)
    y = 10 * n
    autocorr_plot(n, y, name="test", save=True)
    assert os.path.isfile("test_autocorr.png")
    os.remove("test_autocorr.png")
