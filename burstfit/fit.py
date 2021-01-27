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
        keys = self.sgram_params.keys()
        if "all" in keys:
            n = len(keys) - 1
        else:
            n = len(keys)
        return n

    @property
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

    @property
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

    @property
    def make_spectra(self):
        try:
            mu_idx = np.where(np.array(self.profile_param_names) == "mu")[0]
            sig_idx = np.where(np.array(self.profile_param_names) == "sigma")[0]
            assert len(mu_idx) == 1, "mu not found in profile parameter names"
            assert len(sig_idx) == 1, "sigma not found in profile parameter names"
            self.i0 = int(self.profile_params[self.comp_num]["popt"][mu_idx[0]])
            width = int(2.355 * self.profile_params[self.comp_num]["popt"][sig_idx[0]])
        except (KeyError, AssertionError) as e:
            logger.warning(f"{e}")
            width = self.width
            if self.comp_num == 1:
                logger.warning(
                    f"Making spectra using center bins. Could be inaccurate."
                )
                self.i0 = self.nt // 2
            else:
                logger.warning(
                    f"Making spectra using profile argmax. Could be inaccurate."
                )
                self.i0 = np.argmax(self.ts)

        if width > 2:
            start = self.i0 - width // 2
            end = self.i0 + width // 2
        else:
            start = self.i0 - 1
            end = self.i0 + 1

        logging.debug(f"Generating spectra from sample {start} to {end}")
        self.spectra = self.residual[:, start:end].mean(-1)

        if self.comp_num == 1:
            logging.debug(f"Component number is 1. Normalising spectra to unit area.")
            self.spectra = self.spectra / np.trapz(self.spectra)

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
        try:
            popt, pcov = curve_fit(
                self.sgram_model.evaluate,
                xdata=[0],
                ydata=self.residual.ravel(),
                p0=p0,
                bounds=bounds,
            )
        except RuntimeError as e:
            retry_frac = 0.5
            logging.warning(f"{e}")
            logging.warning(f"Retrying with p0+-({retry_frac}*p0) bounds")
            bounds = (np.array(p0) * (1 - retry_frac), np.array(p0) * (1 + retry_frac))
            popt, pcov = curve_fit(
                self.sgram_model.evaluate,
                xdata=[0],
                ydata=self.residual.ravel(),
                p0=p0,
                bounds=bounds,
            )

        err = np.sqrt(np.diag(pcov))
        retry_frac = 0.2
        if np.isinf(err).sum() > 0:
            logging.warning(
                f"Fit errors are not finite. Retrying with p0+-({retry_frac}*p0) bounds"
            )
            bounds = (np.array(p0) * (1 - retry_frac), np.array(p0) * (1 + retry_frac))
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

    def fitcycle(self, plot, sgram_bounds=[-np.inf, np.inf]):
        """

        Args:
            plot:

        Returns:

        """
        logging.info(f"Fitting component {self.comp_num}.")
        self.validate
        self.precalc
        self.initial_profilefit(plot=plot)
        self.make_spectra
        self.initial_spectrafit(plot=plot)
        self.sgram_fit(plot=plot, bounds=sgram_bounds)

    def fit_all_components(self, plot):
        """

        Args:
            plot:

        Returns:

        """
        logging.info(f"Fitting {self.ncomponents} components together.")
        #         self.validate()
        #         self.precalc()

        p0 = []
        for k in self.sgram_params.keys():
            p0 += self.sgram_params[k]["popt"]

        popt, pcov = curve_fit(
            self.model_from_params,
            xdata=[0],
            ydata=self.sgram.ravel(),
            p0=p0,
        )
        err = np.sqrt(np.diag(pcov))
        assert np.isinf(err).sum() == 0, "Errors are not finite. Terminating."

        self.sgram_params["all"] = {"popt": list(popt), "perr": err}
        nparams = len(self.param_names)
        logging.info(f"Converged parameters are:")
        for i in range(self.ncomponents):
            logging.info(f"Component {i}")
            params = self.sgram_params["all"]["popt"][i * nparams : (i + 1) * nparams]
            for j, p in enumerate(params):
                logging.info(f"{self.param_names[j]}: {p} +- {err[j]}")

        if plot:
            plot_2d_fit(
                self.sgram,
                self.model_from_params,
                self.sgram_params["all"]["popt"],
            )

        self.residual = self.sgram - self.model

    def fit_components(self, plot=True, max_ncomp=5, sgram_bounds=[-np.inf, np.inf]):
        """

        Args:
            plot:
            max_ncomp:

        Returns:

        """
        self.precalc
        test_res = self.run_tests
        if test_res:
            logging.warning(
                "On pulse region looks like noise. Check candidate parameters"
            )

        while self.ncomponents < max_ncomp:
            self.fitcycle(plot, sgram_bounds)
            test_res = self.run_tests
            if test_res:
                logging.info(
                    "On pulse residual looks like noise. "
                    "Terminating individual component fitting."
                )
                break
            self.comp_num += 1

        if self.comp_num == max_ncomp:
            logging.info(
                "Max number of components reached. "
                "Terminating individual component fitting."
            )

        if self.ncomponents > 1:
            logging.info(
                f"Final number of components {self.ncomponents} > 1. "
                "Fitting all components together."
            )
            self.fit_all_components()
            test_res = self.run_tests
            if test_res:
                logging.info("On pulse residual looks like noise. ")
            else:
                logging.warning(
                    "On pulse residual doesnot look like noise."
                    "Check the fitting results carefully."
                )
        else:
            logging.info(f"Final number of components = 1. Terminating fitting.")

    @property
    def run_tests(self):
        """

        Returns:

        """
        logging.info(f"Running statistical tests on the residual.")
        on_pulse = self.residual[:, self.i0 - self.width : self.i0 + self.width]
        off_pulse_left = self.sgram[:, 0 : 2 * self.width]
        off_pulse_right = self.sgram[:, -2 * self.width :]
        logging.info("Running off pulse - off pulse test")
        off_off = tests(off_pulse_left, off_pulse_right, ntest=2)

        if off_off == 0:
            logging.warning(f"Off pulse regions are not similar")

        logging.info("Running on pulse - off pulse (L) test")
        ofl_on = tests(off_pulse_left, on_pulse, ntest=2)
        if ofl_on == 1:
            logging.info("On pulse residual is similar to left off pulse region.")

        logging.info("Running on pulse - off pulse (R) test")
        ofr_on = tests(off_pulse_right, on_pulse, ntest=2)
        if ofr_on == 1:
            logging.info("On pulse residual is similar to right off pulse region.")

        return np.any([ofl_on, ofr_on])

    @property
    def model(self):
        """

        Returns:

        """
        logging.info(f"Making model.")
        if "all" in self.sgram_params.keys():
            model = self.model_from_params(
                [0], self.sgram_params["all"]["popt"]
            ).reshape(self.nf, self.nt)
        else:
            assert len(self.sgram_params) == self.ncomponents
            logging.info(f"Found {self.ncomponents} components.")

            model = np.zeros(shape=(self.nf, self.nt))
            for i in range(1, self.ncomponents + 1):
                popt = self.sgram_params[i]["popt"]
                self.sgram_model.forfit = False
                model += self.sgram_model.evaluate([0], *popt)
        return model

    def model_from_params(self, x, params=None):
        """

        Returns:

        """
        logging.info(f"Making model.")
        assert len(params) % len(self.param_names) == 0
        ncomp = int(len(params) / len(self.param_names))
        nparams = int(len(self.param_names))
        logging.info(f"Found {ncomp} components.")

        model = np.zeros(shape=(self.nf, self.nt))
        for i in range(1, ncomp + 1):
            popt = params[(i - 1) * nparams : i * nparams]
            self.sgram_model.forfit = False
            model += self.sgram_model.evaluate([0], *popt)
        return model.ravel()

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
