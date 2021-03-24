import os

import numpy as np
import pytest

from burstfit.data import BurstData
from burstfit.fit import BurstFit
from burstfit.io import BurstIO
from burstfit.model import Model, SgramModel
from burstfit.utils.functions import pulse_fn, gauss_norm2, sgram_fn

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
    pm = Model(pulse_fn, param_names=["S", "mu_t", "sigma_t", "tau"])
    sm = Model(gauss_norm2, param_names=["mu1", "sig1", "mu2", "sig2", "amp1"])
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
    bf.fitall(spectra_bounds=([50, 5, 200, 5, 0], [150, 50, 300, 50, 1]), plot=False)
    return bf


@pytest.fixture(scope="module", autouse=True)
def bio():
    ref_json = os.path.join(_install_dir, "data/test.json")
    bio = BurstIO(jsonfile=ref_json)
    bio.read_json_and_precalc()
    return bio


def compare_res_dict(ref, d):
    for k in ref.keys():
        if "fileheader" in k:
            for kk in ref[k].keys():
                if "file" not in kk:
                    if isinstance(ref[k][kk], float):
                        assert (
                            pytest.approx(ref[k][kk], rel=1e-5 * ref[k][kk]) == d[k][kk]
                        )
                    else:
                        assert ref[k][kk] == d[k][kk]

        elif "_params" in k:
            ref_d = ref[k]["1"].copy()
            d_d = d[k]["1"].copy()
            # for kk in ref_d.keys():
            kk = "popt"
            ref_d[kk] = np.array(ref_d[kk])
            d_d[kk] = np.array(d_d[kk])
            for i, r in enumerate(ref_d[kk]):
                tol = np.max([1e-4, ref_d["perr"][i]])
                assert pytest.approx(r, rel=tol) == d_d[kk][i]
        else:
            assert ref[k] == d[k]


def test_ioclass(bf, bd):
    assert BurstIO(bf, bd)


def test_set_attributes(bf, bd):
    bio = BurstIO(bf, bd)
    bio.set_attributes_to_save()
    assert bio.sgram_params == bf.sgram_params
    assert bio.ncomponents == 1
    assert bio.sgram_function == "sgram_fn"
    assert bio.pulse_function == "pulse_fn"
    assert bio.spectra_function == "gauss_norm2"
    assert bio.fileheader == vars(bd.your_header)
    assert bio.param_names == [
        "mu1",
        "sig1",
        "mu2",
        "sig2",
        "amp1",
        "S",
        "mu_t",
        "sigma_t",
        "tau",
        "DM",
    ]


def test_save_results(bf, bd):
    bio = BurstIO(bf, bd)
    d = bio.save_results(outname="temp.json")
    assert os.path.isfile("temp.json")
    assert d["sgram_params"] == bf.sgram_params
    assert d["ncomponents"] == bf.ncomponents
    assert d["fileheader"] == vars(bd.your_header)
    assert d["profile_bounds"] == bf.profile_bounds
    assert d["spectra_bounds"] == bf.spectra_bounds
    assert d["sgram_bounds"] == bf.sgram_bounds
    assert d["clip_fac"] == bf.clip_fac
    assert d["nt"] == bf.nt
    assert d["nf"] == bf.nf
    assert d["i0"] == bf.i0
    os.remove("temp.json")


def test_read_json_and_precalc(bf, bd, bio):
    bio_params = bio.sgram_params["all"]["1"]["popt"]
    bio_err = bio.sgram_params["all"]["1"]["perr"]

    real_params = [
        74.59582235949433,
        27.97342598198541,
        281.8173333861384,
        45.422769230700155,
        0.4118904298590064,
        562.1550933825228,
        78.93436603779134,
        0.7834801657548776,
        0.07015154728496703,
        474.6979323912283,
    ]
    real_errors = [
        3.0272239357503827,
        2.9417970129387547,
        5.532359257989639,
        5.157774342176106,
        0.03429010794926379,
        20.954648557177954,
        0.05268036879010988,
        0.009891316574292881,
        0.0003835808494783465,
        0.118093058085362,
    ]
    assert bio_params == real_params
    assert bio_err == real_errors

    assert bio.ncomponents == 1
    assert bio.sgram_function == "sgram_fn"
    assert bio.pulse_function == "pulse_fn"
    assert bio.spectra_function == "gauss_norm2"

    header = vars(bd.your_header)
    for key, value in bio.fileheader.items():
        if "base" not in key and "file" not in key and "dtype" not in key:
            assert pytest.approx(value) == header[key]
        elif "dtype" in key:
            assert value == header[key].__name__
    assert bio.clip_fac == bf.clip_fac
    assert bio.nt == bf.nt
    assert bio.nf == bf.nf
    assert bio.i0 == bf.i0


def test_model(bf, bio):
    bf.sgram_model.forfit = False
    assert (
        pytest.approx(
            (
                bf.model_from_params([0], *bio.sgram_params["all"]["1"]["popt"])
                - bio.model
            ).mean(),
            rel=0.1,
        )
        == 0
    )
