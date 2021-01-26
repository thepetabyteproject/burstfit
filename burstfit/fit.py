#!/usr/bin/env python3

import logging

import numpy as np
from scipy.optimize import curve_fit

from burstfit.utils.math import tests, fma
from burstfit.utils.plotter import plot_1d_fit, plot_2d_fit

logger = logging.getLogger(__name__)


class BurstFit:
    """
    BurstFit class to perform spectro-temporal modeling on the burst data

    Args:
        sgram_model:
        sgram:
        width:
        dm:
        foff:
        fch1:
        tsamp:
        clip_fac:
        outname:
    """

    def __init__(
            self,
            sgram_model=None,
            sgram=None,
            width=None,
            dm=None,
            foff=None,
            fch1=None,
            tsamp=None,
            clip_fac=None,
            outname=None,
    ):
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
        self.outname = outname
        self.nt = None
        self.nf = None
        self.i0 = None
        self.ts = None
        self.profile_params = {}
        self.spectra_params = {}
        self.sgram_params = {}
        self.spectra = None
        self.profile_param_names = None
        self.spectra_param_names = None
        self.param_names = None
        self.metadata = None

    @property
    def ncomponents(self):
        """

        Returns: number of components

        """
        return len(self.spectra_params)

    def validate(self):
        """

        Returns:

        """
        assert np.any(self.sgram), "Attribute sgram not set"
        assert self.sgram_model, "Spectrogram model not set"
        assert self.width, "Width not set"
        assert self.dm, "DM not set"
        assert self.foff, "foff not set"
        assert self.fch1, "fch1 not set"
        assert self.tsamp, "tsamp not set"

    def precalc(self):
        """

        Returns:

        """
        logging.debug(f"Running precalculations for component: {self.comp_num}")
        self.nf, self.nt = self.sgram.shape
        assert self.comp_num > 0
        if self.comp_num == 1:
            self.residual = self.sgram
            self.i0 = self.nt // 2
            self.ts = self.residual.sum(0)
        else:
            self.ts = self.residual.sum(0)
            self.i0 = np.argmax(self.ts)

        if self.width > 4:
            self.spectra = self.residual[
                           :, self.i0 - self.width // 4: self.i0 + self.width // 4
                           ].mean(-1)
        else:
            self.spectra = self.residual[:, self.i0 - 1: self.i0 + 1].mean(-1)

        if self.comp_num == 1:
            logging.debug(f"Component number is 1. Normalising spectra to unit area.")
            self.spectra = self.spectra / np.trapz(self.spectra)
        self.profile_param_names = self.sgram_model.pulse_model.param_names
        self.spectra_param_names = self.sgram_model.spectra_model.param_names
        self.param_names = self.sgram_model.param_names
        self.metadata = (
            self.nt,
            self.nf,
            self.dm,
            self.tsamp,
            self.fch1,
            self.foff,
            self.clip_fac,
        )
        self.sgram_model.metadata = self.metadata
        self.sgram_model.forfit = True

    def initial_profilefit(self, plot=False, bounds=[]):
        """

        Args:
            plot:
            bounds:

        Returns:

        """
        logging.info(f"Running initial profile fit for component: {self.comp_num}")
        xdata = np.arange(self.nt)
        ydata = self.ts
        if not np.any(bounds):
            if self.sgram_model.pulse_model.nparams == 4:
                s_bound = 4 * np.trapz(self.ts)
                if s_bound < 0:
                    s_bound = 10 * np.max(self.ts)
                lim = np.min([4 * self.width, self.nt // 2])
                bounds = (
                    [0, self.i0 - lim, 0, 0],
                    [s_bound, self.i0 + lim, lim, lim],
                )
            else:
                bounds = [-np.inf, np.inf]
        logging.debug(f"Bounds for profile fit are: {bounds}")
        popt, pcov = curve_fit(
            self.sgram_model.pulse_model.function,
            xdata,
            ydata,
            bounds=bounds,
        )
        err = np.sqrt(np.diag(pcov))
        assert np.isinf(err).sum() == 0, "Fit errors are not finite. Terminating."
        self.profile_params[self.comp_num] = {"popt": list(popt), "perr": err}

        logging.info(f"Converged parameters (profile fit) are:")
        for i, p in enumerate(self.profile_params[self.comp_num]["popt"]):
            logging.info(f"{self.profile_param_names[i]}: {p} +- {err[i]}")

        if plot:
            plot_1d_fit(
                xdata,
                ydata,
                self.sgram_model.pulse_model.function,
                self.profile_params[self.comp_num]["popt"],
                xlabel="Time",
                ylabel="Amp",
                title="Initial fit to profile",
                param_names=self.profile_param_names,
            )

    def initial_spectrafit(self, plot=False, bounds=[]):
        """

        Args:
            plot:
            bounds:

        Returns:

        """
        logging.info(f"Running spectra profile fit for component: {self.comp_num}")
        xdata = np.arange(self.nf)
        ydata = self.spectra
        if not np.any(bounds):
            if self.sgram_model.spectra_model.nparams == 2:
                bounds = ([xdata.min(), 0], [xdata.max(), xdata.max()])
            else:
                bounds = [-np.inf, np.inf]
        logging.debug(f"Bounds for spectra fit are: {bounds}")
        popt, pcov = curve_fit(
            self.sgram_model.spectra_model.function,
            xdata,
            ydata,
            bounds=bounds,
        )
        err = np.sqrt(np.diag(pcov))
        assert np.isinf(err).sum() == 0, "Fit errors are not finite. Terminating."
        self.spectra_params[self.comp_num] = {"popt": list(popt), "perr": err}

        logging.info(f"Converged parameters (spectra fit) are:")
        for i, p in enumerate(self.spectra_params[self.comp_num]["popt"]):
            logging.info(f"{self.spectra_param_names[i]}: {p} +- {err[i]}")

        if plot:
            plot_1d_fit(
                xdata,
                ydata,
                self.sgram_model.spectra_model.function,
                self.spectra_params[self.comp_num]["popt"],
                xlabel="Channels",
                ylabel="Amp",
                title="Initial fit to spectra",
                param_names=self.spectra_param_names,
            )

    def sgram_fit(self, plot=False, bounds=[-np.inf, np.inf]):
        """

        Args:
            plot:
            bounds:

        Returns:

        """
        logging.info(f"Running sgram profile fit for component: {self.comp_num}")
        p0 = (
                self.spectra_params[self.comp_num]["popt"]
                + self.profile_params[self.comp_num]["popt"]
                + [self.dm]
        )
        popt, pcov = curve_fit(
            self.sgram_model.evaluate,
            xdata=[0],
            ydata=self.residual.ravel(),
            p0=p0,
            bounds=bounds,
        )
        err = np.sqrt(np.diag(pcov))
        if np.isinf(err).sum() > 0:
            logging.warning(
                "Fit errors are not finite. Retrying with p0+-(0.2*p0) bounds"
            )
            bounds = (np.array(p0) * 0.8, np.array(p0) * 1.2)
            popt, pcov = curve_fit(
                self.sgram_model.evaluate,
                xdata=[0],
                ydata=self.residual.ravel(),
                p0=p0,
                bounds=bounds,
            )
            err = np.sqrt(np.diag(pcov))
            assert np.isinf(err).sum() == 0, "Errors are still not finite. Terminating."

        self.sgram_params[self.comp_num] = {"popt": list(popt), "perr": err}
        logging.info(f"Converged parameters are:")
        for i, p in enumerate(self.sgram_params[self.comp_num]["popt"]):
            logging.info(f"{self.param_names[i]}: {p} +- {err[i]}")

        self.sgram_model.forfit = False
        if plot:
            plot_2d_fit(
                self.residual,
                self.sgram_model.evaluate,
                self.sgram_params[self.comp_num]["popt"],
            )

        self.residual = self.sgram - self.model

    def fitcycle(self, plot):
        """

        Args:
            plot:

        Returns:

        """
        logging.info(f"Fitting component {self.comp_num}.")
        self.validate()
        self.precalc()
        self.initial_profilefit(plot=plot)
        self.initial_spectrafit(plot=plot)
        self.sgram_fit(plot=plot)

    def fitall(self, plot=True, max_ncomp=5):
        """

        Args:
            plot:
            max_ncomp:

        Returns:

        """
        self.precalc()
        test_res = self.run_tests
        if test_res:
            logging.warning(
                "On pulse region looks like noise. Check candidate parameters"
            )

        while self.ncomponents < max_ncomp:
            self.fitcycle(plot)
            test_res = self.run_tests
            if test_res:
                logging.info("On pulse residual looks like noise. Terminating fitting.")
                break
            self.comp_num += 1

        if self.comp_num == max_ncomp:
            logging.info("Max number of components reached. Terminating fitting.")

    @property
    def run_tests(self):
        """

        Returns:

        """
        logging.info(f"Running statistical tests on the residual.")
        on_pulse = self.residual[:, self.i0 - self.width: self.i0 + self.width]
        off_pulse_left = self.sgram[:, 0: 2 * self.width]
        off_pulse_right = self.sgram[:, -2 * self.width:]
        logging.info("Running off pulse - off pulse test")
        off_off = tests(off_pulse_left, off_pulse_right)

        if off_off == 0:
            logging.warning(f"Off pulse regions are not similar")

        logging.info("Running on pulse - off pulse (L) test")
        ofl_on = tests(off_pulse_left, on_pulse)
        if ofl_on == 1:
            logging.info("On pulse residual is similar to left off pulse region.")

        logging.info("Running on pulse - off pulse (R) test")
        ofr_on = tests(off_pulse_right, on_pulse)
        if ofr_on == 1:
            logging.info("On pulse residual is similar to right off pulse region.")

        return np.any([ofl_on, ofr_on])

    @property
    def model(self):
        """

        Returns:

        """
        logging.info(f"Making model.")
        assert len(self.sgram_params) == self.ncomponents
        logging.info(f"Found {self.ncomponents} components.")

        model = np.zeros(shape=(self.nf, self.nt))
        for i in range(1, self.ncomponents + 1):
            popt = self.sgram_params[i]["popt"]
            self.sgram_model.forfit = False
            model += self.sgram_model.evaluate([0], *popt)
        return model

    def get_physical_params(self, mapping, params=None, errors=None):
        """

        Args:
            mapping:
            params:
            errors:

        Returns:

        """
        if not params:
            params = self.sgram_params[self.comp_num]["popt"]

        if not errors:
            errors = self.sgram_params[self.comp_num]["perr"]

        assert len(mapping) == len(params)
        assert len(mapping) == len(errors)

        physical_dict = {}
        physical_errs = {}
        for key in mapping:
            k, m, a = mapping[key]
            param = params[self.param_names.index(k)]
            physical_dict[key] = fma(param, m, a)
            err = errors[self.param_names.index(k)]
            physical_errs[key] = fma(err, m, 0)
        return physical_dict, physical_errs
