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
    bf.fitall(spectra_bounds=([50, 5, 200, 5, 0], [150, 50, 300, 50, 1]), plot=False)
    return bf


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
    assert bio.ncomponents == 1
    assert bio.sgram_function == "sgram_fn"
    assert bio.pulse_function == "pulse_fn"
    assert bio.spectra_function == "gauss_norm2"


def test_read_json_and_precalc(bf):
    ref_json = os.path.join(_install_dir, "data/28.json")
    bio = BurstIO(jsonfile=ref_json)


#  bio.read_json_and_precalc()
#  assert bio.sgramModel.metadata == bf.metadata
#  assert bio.sgramModel.param_names == bf.param_names


# def test_save_results(bf, bd):
#     bio = BurstIO(bf, bd)
#     bio.save_results(outname='test.json')
#     assert os.path.isfile('test.json')
#
#     with open('test.json', "r") as fp:
#         d = json.load(fp)
#
#     ref_json = os.path.join(_install_dir, "data/28.json")
#     with open(ref_json, "r") as fp:
#         ref = json.load(fp)
#
#     compare_res_dict(ref, d)
#
#     os.remove("test.json")
