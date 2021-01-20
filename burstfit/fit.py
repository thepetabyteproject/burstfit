#!/usr/bin/env python3

import numpy as np
from burstfit.utils.plotter import plot_1d_fit, plot_2d_fit
from burstfit.utils.models import sgram_model
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
        clip_frac=None,
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
        self.ncomps = 1
        self.profile_params = {}
        self.spectra_params = {}
        self.sgram_params = {}
        self.clip_frac = clip_frac
        self.residual = None

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
        if not np.any(self.residual):
            self.residual = self.sgram
            self.i0 = self.nt // 2
            self.ts = self.residual.sum(0)
        else:
            self.ts = self.residual.sum(0)
            self.i0 = np.argmax(self.ts)
        self.spectra = self.residual[
            :, self.i0 - self.width // 4 : self.i0 + self.width // 4
        ].mean(-1)
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
            self.clip_frac,
        )

    def initial_profilefit(self, plot=False, bounds=[]):
        xdata = np.arange(self.nt)
        ydata = self.ts
        if not np.any(bounds):
            bounds = (
                [0, xdata.min(), 0, 0],
                [np.trapz(self.ts), xdata.max(), xdata.max(), xdata.max() // 5],
            )
        popt, pcov = curve_fit(
            self.profile_model,
            xdata,
            ydata,
            bounds=bounds,
        )
        err = np.sqrt(np.diag(pcov))
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

    def make_model(self):
        assert len(self.sgram_params) == self.ncomps
        logging.info(f"Found {self.ncomps} components.")

        model = np.zeros(shape=(self.nf, self.nt))
        for i in range(1, self.ncomps + 1):
            popt = self.sgram_params[i]["popt"]
            model += sgram_model(self.metadata, *popt).reshape(self.nf, self.nt)
        return model
