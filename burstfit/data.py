#!/usr/bin/env python3

import numpy as np
from your.candidate import Candidate
import logging
logger = logging.getLogger(__name__)

class BurstData(Candidate):
    """
    """
    def __init__(
        self,
        fp=None,
        dm=None,
        tcand=0,
        width=0,
        snr=0,
        min_samp=256,
        device=0,
        kill_mask=None,
    ):
        Candidate.__init__(
            self,
            fp=fp,
            dm=dm,
            tcand=tcand,
            width=width,
            label=-1,
            snr=snr,
            min_samp=min_samp,
            device=0,
            kill_mask=None,
        )

    def prepare_data(self, mask_chans=[], time_window=200e-3, normalise=True):
        self.get_chunk()
        nt, nf = self.data.shape
        self.i0 = nt // 2
        self.dedisperse()
        self.dedispersed = np.ma.array(
            self.dedispersed, mask=False * np.ones(self.dedispersed.shape)
        )

        if np.any(mask_chans):
            self.mask_channels(mask_chans)

        self.dispersed_at_dm = self.dm
        self.sgram = self.crop_dedispersed_data(time_window)

        if normalise:
            off_pulse_data = self.dedispersed[
                :, : self.i0 - int(2 * time_window // self.tsamp)
            ]
            self.sgram, self.clip_frac = self.normalise_data(self.sgram, off_pulse_data)
        return self

    def nstart(self):
        nt, nf = self.data.shape
        return self.tcand // self.tsamp - (nt // 2)

    def mask_channels(self, mask_chans=[]):
        for m in mask_chans:
            if isinstance(m, tuple):
                self.dedispersed.mask[:, m[0] : m[1]] = True
            elif isinstance(m, int):
                self.dedispersed.mask[:, m] = True
            else:
                raise AttributeError(
                    "mask_chans can only contain tuple (start_chan:end_chan) and/or ints"
                )
        return self

    def normalise_data(self, on_pulse_data, off_pulse_data, return_clip_frac=True):
        off_pulse_mean = np.mean(off_pulse_data)
        off_pulse_std = np.std(off_pulse_data)
        logging.info(f"Off pulse mean and std are: {off_pulse_mean, off_pulse_std}")
        on_pulse_data = on_pulse_data - off_pulse_mean
        on_pulse_data = on_pulse_data / off_pulse_std
        if return_clip_frac:
            clip_frac = ((2 ** self.nbits - 1) - off_pulse_mean) / off_pulse_std
        return on_pulse_data, clip_frac

    def crop_dedispersed_data(self, time_window):
        time_around_burst = int(time_window // self.tsamp // 2)
        return self.dedispersed[
            self.i0 - time_around_burst : self.i0 + time_around_burst, :
        ].T