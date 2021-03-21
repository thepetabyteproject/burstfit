#!/usr/bin/env python3

import logging

import numpy as np
from your.candidate import Candidate

logger = logging.getLogger(__name__)


class BurstData(Candidate):
    """
    Class to handle burst data

    Args:
        fp (Union[str, list]): String or a list of files. It can either filterbank or psrfits files.
        dm (float): Dispersion Measure of the candidate
        tcand (float): start time of the candidate in seconds at the highest frequency channel
        width (int): pulse width of the candidate in samples
        snr (float): Signal to Noise Ratio
        min_samp (int): Minimum number of time samples
        kill_mask (numpy.ndarray): Boolean mask of channels to kill
        spectral_kurtosis_sigma (float): Sigma for spectral kurtosis filter
        savgol_frequency_window (float): Filter window for savgol filter
        savgol_sigma (float):  Sigma for savgol filter
        flag_rfi (bool): To turn on RFI flagging
    """

    def __init__(
        self,
        fp=None,
        dm=None,
        tcand=0,
        width=0,
        snr=0,
        min_samp=256,
        kill_mask=np.array([False]),
        spectral_kurtosis_sigma=4,
        savgol_frequency_window=15,
        savgol_sigma=4,
        flag_rfi=False,
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
            kill_mask=kill_mask,
            spectral_kurtosis_sigma=spectral_kurtosis_sigma,
            savgol_frequency_window=savgol_frequency_window,
            savgol_sigma=savgol_sigma,
            flag_rfi=flag_rfi,
        )
        self.dispersed_at_dm = None
        self.i0 = None
        self.clip_fac = None
        self.sgram = None
        self.input_mask = np.zeros(self.nchans, dtype="bool")

    def prepare_data(self, mask_chans=[], time_window=200e-3, normalise=True):
        """
        Prepares data for burst fitting

        Args:
            mask_chans: list with tuples (start_freq, end_freq) and channel numbers to mask
            time_window: time window (s) around the burst to use for burst fitting
            normalise: To normalise the mean and std of the data using an off pulse region

        Returns:

        """
        logger.info("Preparing data for burst fitting.")
        self.get_chunk()
        nt, nf = self.data.shape
        self.i0 = nt // 2
        self.dedisperse()
        self.dedispersed = np.ma.array(
            self.dedispersed, mask=False * np.ones(self.dedispersed.shape)
        )

        if np.any(mask_chans):
            self.prepare_input_mask(mask_chans)

        self.mask_channels()

        self.dispersed_at_dm = self.dm
        self.sgram = self.crop_dedispersed_data(time_window)

        if normalise:
            off_pulse_data = self.dedispersed[
                :, : self.i0 - int(2 * time_window // self.tsamp)
            ]
            self.sgram, self.clip_fac = self.normalise_data(self.sgram, off_pulse_data)
        return self

    @property
    def nstart(self):
        """

        Returns: start sample number of the spectrogram

        """
        nt, nf = self.sgram.shape
        return self.tcand // self.tsamp - (nt // 2)

    @property
    def mask(self):
        """

        Returns: Channel mask array using all the available masks

        """
        m = self.input_mask
        if self.kill_mask.any():
            m = self.kill_mask | m
        if self.rfi_mask.any():
            m = self.rfi_mask | m
        return m

    def prepare_input_mask(self, mask_chans=[]):
        """
        Function to mask some frequency channels using input_mask, kill_mask and rfi_mask

        Args:
            mask_chans: list with tuples (start_freq, end_freq) and channel numbers to mask

        Returns:

        """
        logger.debug(f"Preparing input mask.")
        for m in mask_chans:
            if isinstance(m, tuple):
                assert len(m) == 2
                self.input_mask[m[0] : m[1]] = True
            elif isinstance(m, int):
                self.input_mask[m] = True
            elif isinstance(m, list):
                assert len(m) == 2
                self.input_mask[m[0] : m[1]] = True
            else:
                raise AttributeError(
                    "mask_chans can only contain tuple/list (start_chan:end_chan) and/or ints"
                )
        return self

    def mask_channels(self):
        """
        Apply channel  mask to the dedispersed data

        Returns:

        """
        logger.debug("Masking channels.")
        self.dedispersed.mask[:, self.mask] = True
        return self

    def normalise_data(self, on_pulse_data, off_pulse_data, return_clip_fac=True):
        """
        Function to normalise data

        Args:
            on_pulse_data: Data to normalise
            off_pulse_data: Data to use to estimate mean and std
            return_clip_fac: To return the clipping factor, decided using nbits of data

        Returns:

        """
        logger.info(f"Normalising data using off pulse mean and std.")
        off_pulse_mean = np.mean(off_pulse_data)
        off_pulse_std = np.std(off_pulse_data)
        logger.info(f"Off pulse mean and std are: {off_pulse_mean, off_pulse_std}")
        on_pulse_data = on_pulse_data - off_pulse_mean
        on_pulse_data = on_pulse_data / off_pulse_std
        if return_clip_fac:
            clip_fac = ((2 ** self.nbits - 1) - off_pulse_mean) / off_pulse_std
            logger.debug(f"Clip factor is {clip_fac}")
            return on_pulse_data, clip_fac
        else:
            return on_pulse_data

    def crop_dedispersed_data(self, time_window):
        """

        To get a cutout of data from only around the burst

        Args:
            time_window: time length to use on both sides of burst for the cutout

        Returns:

        """
        logger.info(f"Cropping data with time_window: {time_window}s.")
        time_around_burst = int(time_window // self.tsamp // 2)
        return self.dedispersed[
            self.i0 - time_around_burst : self.i0 + time_around_burst, :
        ].T
