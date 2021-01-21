#!/usr/bin/env python3

import numpy as np
from burstfit.utils.plotter import plot_1d_fit, plot_2d_fit
from burstfit.utils.models import sgram_model
from burstfit.utils.math import tests
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)


class BurstFit:
    def __init__(
        self,
        profile_model=None,
        spectra_model=None,
        sgram_model=None,
        sgram=None,
        width=None,
        dm=None,
        foff=None,
        fch1=None,
        tsamp=None,
        clip_fac=None,
    ):
        self.profile_model = profile_model
        self.spectra_model = spectra_model
        self.sgram_model = sgram_model
        self.sgram = sgram
        self.width = width
        self.dm = dm
        self.foff = foff
        self.fch1 = fch1
        self.tsamp = tsamp
        self.comp_num = 1
        self.profile_params = {}
        self.spectra_params = {}
        self.sgram_params = {}
        self.clip_fac = clip_fac
        self.residual = None

    @property
    def ncomponents(self):
        return len(self.spectra_params)

    def validate(self):
        assert np.any(self.sgram), "Attribute sgram not set"
        assert self.profile_model, "Profile model not set"
        assert self.spectra_model, "Spectra model not set"
        assert self.sgram_model, "Spectrogram model not set"
        assert self.width, "Width not set"
        assert self.dm, "DM not set"
        assert self.foff, "foff not set"
        assert self.fch1, "fch1 not set"
        assert self.tsamp, "tsamp not set"

    def precalc(self):
        self.nf, self.nt = self.sgram.shape
        assert self.comp_num > 0
        if self.comp_num == 1:
            self.residual = self.sgram
            self.i0 = self.nt // 2
            self.ts = self.residual.sum(0)
        else:
            self.ts = self.residual.sum(0)
            self.i0 = np.argmax(self.ts)

        self.spectra = self.residual[
            :, self.i0 - self.width // 4 : self.i0 + self.width // 4
        ].mean(-1)
        self.spectra = self.spectra / np.trapz(self.spectra)
        self.profile_param_names = ["S_t", "t_mu", "t_sigma", "tau"]
        self.spectra_param_names = ["nu_0", "nu_sig"]
        self.param_names = self.spectra_param_names + self.profile_param_names + ["DM"]
        self.metadata = (
            self.nt,
            self.nf,
            self.dm,
            self.tsamp,
            self.fch1,
            self.foff,
            self.clip_fac,
        )

    def initial_profilefit(self, plot=False, bounds=[]):
        xdata = np.arange(self.nt)
        ydata = self.ts
        if not np.any(bounds):
            s_bound = np.max([np.trapz(self.ts), 40 * np.max(self.ts)])
            lim = np.min([4 * self.width, self.nt // 2])
            bounds = (
                [0, self.i0 - lim, 0, 0],
                [s_bound, self.i0 + lim, lim, lim],
            )
        logging.debug(f"Bounds for profile fit are: {bounds}")
        popt, pcov = curve_fit(
            self.profile_model,
            xdata,
            ydata,
            bounds=bounds,
        )
        err = np.sqrt(np.diag(pcov))
        assert np.isinf(err).sum() == 0, "Something wrong with the profile fit"
        self.profile_params[self.comp_num] = {"popt": list(popt), "perr": err}

        if plot:
            plot_1d_fit(
                xdata,
                ydata,
                self.profile_model,
                self.profile_params[self.comp_num]["popt"],
                xlabel="Time",
                ylabel="Amp",
                title="Initial fit to profile",
                param_names=self.profile_param_names,
            )

    def initial_spectrafit(self, plot=False):
        xdata = np.arange(self.nf)
        ydata = self.spectra
        popt, pcov = curve_fit(
            self.spectra_model,
            xdata,
            ydata,
            bounds=([xdata.min(), 0], [xdata.max(), xdata.max()]),
        )
        err = np.sqrt(np.diag(pcov))
        assert np.isinf(err).sum() == 0, "Something wrong with the spectra fit"

        self.spectra_params[self.comp_num] = {"popt": list(popt), "perr": err}
        if plot:
            plot_1d_fit(
                xdata,
                ydata,
                self.spectra_model,
                self.spectra_params[self.comp_num]["popt"],
                xlabel="Channels",
                ylabel="Amp",
                title="Initial fit to spectra",
                param_names=self.spectra_param_names,
            )

    def sgram_fit(self, plot=False):
        p0 = (
            self.spectra_params[self.comp_num]["popt"]
            + self.profile_params[self.comp_num]["popt"]
            + [self.dm]
        )
        popt, pcov = curve_fit(
            self.sgram_model,
            xdata=self.metadata,
            ydata=self.residual.ravel(),
            p0=p0,
        )
        err = np.sqrt(np.diag(pcov))
        assert np.isinf(err).sum() == 0, "Something wrong with the sgram fit"

        self.sgram_params[self.comp_num] = {"popt": list(popt), "perr": err}
        logging.info(f"Converged parameters are:")

        for i, p in enumerate(self.sgram_params[self.comp_num]["popt"]):
            logging.info(f"{self.param_names[i]}: {p} +- {err[i]}")

        if plot:
            plot_2d_fit(
                self.metadata,
                self.residual,
                self.sgram_model,
                self.sgram_params[self.comp_num]["popt"],
            )

        self.residual = self.sgram - self.make_model()

    def fitcycle(self, plot):
        self.validate()
        self.precalc()
        self.initial_profilefit(plot=plot)
        self.initial_spectrafit(plot=plot)
        self.sgram_fit(plot=plot)

    def fitall(self, plot=True, max_ncomp=5):
        self.precalc()
        test_res = self.run_tests()
        if test_res:
            raise "On pulse region looks like noise. Check candidate parameters"

        while self.ncomponents < max_ncomp:
            logging.info(f"Fitting component {self.comp_num}.")
            self.fitcycle(plot)
            test_res = self.run_tests()
            if test_res:
                logging.info("On pulse residual looks like noise. Terminating fitting.")
                break
            self.comp_num += 1

    def run_tests(self):
        on_pulse = self.residual[:, self.i0 - self.width : self.i0 + self.width]
        off_pulse_left = self.sgram[:, 0 : 2 * self.width]
        off_pulse_right = self.sgram[:, -2 * self.width :]
        logging.info("Running off pulse - off pulse test")
        off_off = tests(off_pulse_left, off_pulse_right)

        assert off_off == 1, "Off pulse regions are not similar"

        logging.info("Running on pulse - off pulse (L) test")
        ofl_on = tests(off_pulse_left, on_pulse)
        if ofl_on == 1:
            logging.info("On pulse residual is similar to left off pulse region.")

        logging.info("Running on pulse - off pulse (R) test")
        ofr_on = tests(off_pulse_right, on_pulse)
        if ofr_on == 1:
            logging.info("On pulse residual is similar to right off pulse region.")

        return np.any([ofl_on, ofr_on])

    def make_model(self):
        assert len(self.sgram_params) == self.ncomponents
        logging.info(f"Found {self.ncomponents} components.")

        model = np.zeros(shape=(self.nf, self.nt))
        for i in range(1, self.ncomponents + 1):
            popt = self.sgram_params[i]["popt"]
            model += sgram_model(self.metadata, *popt).reshape(self.nf, self.nt)
        return model
