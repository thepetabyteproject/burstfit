import os

import numpy as np

from burstfit.data import BurstData

_install_dir = os.path.abspath(os.path.dirname(__file__))
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


def test_input_mask_chans(bd):
    bd.prepare_data(mask_chans=[10, 12, [14, 16]], normalise=False)
    assert np.ma.is_masked(bd.sgram[12, :])
    assert np.ma.is_masked(bd.sgram[10, :])
    assert np.ma.is_masked(bd.sgram[15, :])
    assert not np.ma.is_masked(bd.sgram[9, :])
    assert not np.ma.is_masked(bd.sgram[17, :])

    bd.prepare_data(mask_chans=[(1, 4)])
    assert np.ma.is_masked(bd.sgram[1, :])
    assert np.ma.is_masked(bd.sgram[2, :])
    assert np.ma.is_masked(bd.sgram[3, :])


def test_all_masks(bd):
    bd.flag_rfi = True
    bd.kill_mask = np.zeros(bd.nchans, dtype="bool")
    bd.kill_mask[[10, 30, 100]] = True

    bd.prepare_data(mask_chans=[(1, 4)])
    mask_list = [172, 173, 174, 175, 176, 177, 182, 10, 30, 100, 1, 2, 3]

    assert np.ma.is_masked(bd.sgram[mask_list, :])
