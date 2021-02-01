import os

import numpy as np

from burstfit.data import BurstData

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
    return bd


def test_prepare_data(bd):
    bd.prepare_data(time_window=200e-3, normalise=True)
    assert bd.sgram.all()
    nf, nt = bd.sgram.shape
    assert nf == 336
    assert nt == 156
    assert pytest.approx(bd.sgram.mean(), abs=0.1) == 0
    assert pytest.approx(bd.sgram.std(), abs=0.1) == 1
    assert bd.clip_fac


def test_mask_chans(bd):
    bd.prepare_data(mask_chans=[10, 12])
    assert np.ma.is_masked(bd.sgram[12, :])
    assert np.ma.is_masked(bd.sgram[10, :])
    assert not np.ma.is_masked(bd.sgram[9, :])

    bd.prepare_data(mask_chans=[(1, 4)])
    assert np.ma.is_masked(bd.sgram[1, :])
    assert np.ma.is_masked(bd.sgram[2, :])
    assert np.ma.is_masked(bd.sgram[3, :])
