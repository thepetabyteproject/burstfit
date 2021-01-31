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
        sgram_model: Spectrogram Model class object
        sgram: 2D array of spectrogram
        width: width of the candidate
        dm: DM of the candidate
        foff: frequency resolution of the data
        fch1: Frequency of first channel (MHz))
        tsamp: Sampling interval (seconds)
        clip_fac: Clip factor based on nbits of data
        outname: Outname for the outputs
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

    def validate(self):
        """
        Validate the class attributes

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
        Perform precalculations for fitting

        Returns:

        """
        logger.debug(f"Running precalculations for component: {self.comp_num}")
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

    def make_spectra(self):
        """
        Make the spectra by using the profile fitting parameters.

        Returns:

        """
        tau_width = 0
        try:
            logger.info("Making spectra using profile fit parameters.")
            mu_idx = np.where(np.array(self.profile_param_names) == "mu")[0]
            sig_idx = np.where(np.array(self.profile_param_names) == "sigma")[0]
            assert len(mu_idx) == 1, "mu not found in profile parameter names"
            assert len(sig_idx) == 1, "sigma not found in profile parameter names"
            self.i0 = self.profile_params[self.comp_num]["popt"][mu_idx[0]]
            width = 2.355 * self.profile_params[self.comp_num]["popt"][sig_idx[0]]
            if "tau" in self.profile_param_names:
                t_idx = np.where(np.array(self.profile_param_names) == "tau")[0]
                assert len(t_idx) == 1, "tau not found in profile parameter names"
                tau_width += self.profile_params[self.comp_num]["popt"][t_idx[0]]
            width = int(width)
            self.i0 = int(self.i0)
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
        if start < 0:
            start = 0
        if end > self.nt:
            end = self.nt
        end += int(tau_width)
        logger.debug(f"Generating spectra from sample {start} to {end}")
        self.spectra = self.residual[:, start:end].mean(-1)

        logger.debug(f"Normalising spectra to unit area.")
        self.spectra = self.spectra / np.trapz(self.spectra)

    def fitcycle(self, plot, sgram_bounds=[-np.inf, np.inf]):
        """
        Run the fitting cycle to fit one component

        Args:
            plot: To plot
            sgram_bounds: Bounds for spectrogram fitting

        Returns:

        """
        logger.info(f"Fitting component {self.comp_num}.")
        self.validate()
        self.precalc()
        self.initial_profilefit(plot=plot)
        self.make_spectra()
        self.initial_spectrafit(plot=plot)
        self.sgram_fit(plot=plot, bounds=sgram_bounds)

    def initial_profilefit(self, plot=False, bounds=[]):
        """
        Perform initial profile fit on the pulse.

        Args:
            plot: To plot the fit result.
            bounds: Bounds for fitting.

        Returns:

        """
        logger.info(f"Running initial profile fit for component: {self.comp_num}")
        xdata = np.arange(self.nt)
        ydata = self.ts
        if not np.any(bounds):
            if self.sgram_model.pulse_model.nparams == 4:
                s_bound = 4 * np.trapz(self.ts)
                if s_bound <= 0:
                    if np.max(self.ts) > 0:
                        s_bound = 10 * np.max(self.ts)
                    else:
                        s_bound = 1
                lim = np.min([4 * self.width, self.nt // 2])
                bounds = (
                    [0, self.i0 - lim, 0, 0],
                    [s_bound, self.i0 + lim, lim, lim],
                )
            else:
                bounds = [-np.inf, np.inf]
        logger.debug(f"Bounds for profile fit are: {bounds}")
        popt, pcov = curve_fit(
            self.sgram_model.pulse_model.function,
            xdata,
            ydata,
            bounds=bounds,
        )
        err = np.sqrt(np.diag(pcov))
        assert np.isinf(err).sum() == 0, "Fit errors are not finite. Terminating."
        self.profile_params[self.comp_num] = {"popt": list(popt), "perr": err}

        logger.info(f"Converged parameters (profile fit) are:")
        for i, p in enumerate(self.profile_params[self.comp_num]["popt"]):
            logger.info(f"{self.profile_param_names[i]}: {p} +- {err[i]}")

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
        Perform initial spectra fit on the spectra.

        Args:
            plot: To plot the fitting results.
            bounds: Bounds for fitting.

        Returns:

        """
        logger.info(f"Running spectra profile fit for component: {self.comp_num}")
        xdata = np.arange(self.nf)
        ydata = self.spectra
        if not np.any(bounds):
            if self.sgram_model.spectra_model.nparams == 2:
                bounds = ([xdata.min(), 0], [xdata.max(), xdata.max()])
            else:
                bounds = [-np.inf, np.inf]
        logger.debug(f"Bounds for spectra fit are: {bounds}")
        popt, pcov = curve_fit(
            self.sgram_model.spectra_model.function,
            xdata,
            ydata,
            bounds=bounds,
        )
        err = np.sqrt(np.diag(pcov))
        assert np.isinf(err).sum() == 0, "Fit errors are not finite. Terminating."
        self.spectra_params[self.comp_num] = {"popt": list(popt), "perr": err}

        logger.info(f"Converged parameters (spectra fit) are:")
        for i, p in enumerate(self.spectra_params[self.comp_num]["popt"]):
            logger.info(f"{self.spectra_param_names[i]}: {p} +- {err[i]}")

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
        Perform fit on the spectrogram and updates the residual.

        Args:
            plot: To plot the fitting results.
            bounds: Bounds on the spectrogram fit.

        Returns:

        """
        logger.info(f"Running sgram profile fit for component: {self.comp_num}")
        p0 = (
                self.spectra_params[self.comp_num]["popt"]
                + self.profile_params[self.comp_num]["popt"]
                + [self.dm]  # , 4]
        )

        logger.info(f"initial estimate for parameters: {p0}")
        try:
            popt, pcov = curve_fit(
                self.sgram_model.evaluate,
                xdata=[0],
                ydata=self.residual.ravel(),
                p0=p0,
                bounds=bounds,
            )
        except RuntimeError as e:
            retry_frac = 0.9
            logger.warning(f"{e}")
            logger.warning(f"Retrying with p0+-({retry_frac}*p0) bounds")
            p0_1 = np.array(p0) * (1 - retry_frac)
            p0_2 = np.array(p0) * (1 + retry_frac)
            bounds = (np.min([p0_1, p0_2], axis=0), np.max([p0_1, p0_2], axis=0))
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
            logger.warning(
                f"Fit errors are not finite. Retrying with p0+-({retry_frac}*p0) bounds"
            )
            p0_1 = np.array(p0) * (1 - retry_frac)
            p0_2 = np.array(p0) * (1 + retry_frac)
            bounds = (np.min([p0_1, p0_2], axis=0), np.max([p0_1, p0_2], axis=0))
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
        logger.info(f"Converged parameters are:")
        for i, p in enumerate(self.sgram_params[self.comp_num]["popt"]):
            logger.info(f"{self.param_names[i]}: {p} +- {err[i]}")

        self.sgram_model.forfit = False
        if plot:
            plot_2d_fit(
                self.residual,
                self.sgram_model.evaluate,
                self.sgram_params[self.comp_num]["popt"],
            )

        self.residual = self.sgram - self.model

    def fit_all_components(self, plot):
        """
        Fit all components together (used if num_comp > 1)

        Args:
            plot: To plot the fitting results.

        Returns:

        """
        logger.info(f"Fitting {self.ncomponents} components together.")
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
        logger.info(f"Converged parameters are:")
        for i in range(self.ncomponents):
            logger.info(f"Component {i}")
            params = self.sgram_params["all"]["popt"][i * nparams: (i + 1) * nparams]
            for j, p in enumerate(params):
                logger.info(f"{self.param_names[j]}: {p} +- {err[j]}")

        if plot:
            plot_2d_fit(
                self.sgram,
                self.model_from_params,
                self.sgram_params["all"]["popt"],
            )

        self.residual = self.sgram - self.model

    def fitall(self, plot=True, max_ncomp=5, sgram_bounds=[-np.inf, np.inf]):
        """
        Perform spectro-temporal fitting on the spectrogram for all the components.

        Args:
            plot: to plot the fitting results.
            max_ncomp: maximum number of components to fit.
            sgram_bounds: bounds on spectrogram fit.

        Returns:

        """
        self.precalc()
        test_res = self.run_tests
        if test_res:
            logger.warning(
                "On pulse region looks like noise. Check candidate parameters"
            )

        while self.ncomponents < max_ncomp:
            self.fitcycle(plot, sgram_bounds)
            test_res = self.run_tests
            if test_res:
                logger.info(
                    "On pulse residual looks like noise. "
                    "Terminating individual component fitting."
                )
                break
            self.comp_num += 1

            if self.comp_num == max_ncomp:
                logger.info(
                    "Max number of components reached. "
                    "Terminating individual component fitting."
                )

        if self.ncomponents > 1:
            logger.info(
                f"Final number of components {self.ncomponents} > 1. "
                "Fitting all components together."
            )
            self.fit_all_components(plot)
            test_res = self.run_tests
            if test_res:
                logger.info("On pulse residual looks like noise. ")
            else:
                logger.warning(
                    "On pulse residual doesnot look like noise."
                    "Check the fitting results carefully."
                )
        else:
            popt = self.sgram_params[1]["popt"]
            err = self.sgram_params[1]["perr"]
            self.sgram_params["all"] = {"popt": popt, "perr": err}
            logger.info(f"Final number of components = 1. Terminating fitting.")

    @property
    def run_tests(self):
        """
        Run statistical tests to compare ON pulse residual with OFF pulse spectrogram distributions.

        Returns:
            True if either of the left or right OFF pulse regions are similar to the residual ON pulse region.

        """
        logger.info(f"Running statistical tests on the residual.")
        on_pulse = self.residual[:, self.i0 - self.width: self.i0 + self.width]
        off_pulse_left = self.sgram[:, 0: 2 * self.width]
        off_pulse_right = self.sgram[:, -2 * self.width:]
        logger.info("Running off pulse - off pulse test")
        off_off = tests(off_pulse_left, off_pulse_right, ntest=2)

        if off_off == 0:
            logger.warning(f"Off pulse regions are not similar")

        logger.info("Running on pulse - off pulse (L) test")
        ofl_on = tests(off_pulse_left, on_pulse, ntest=2)
        if ofl_on == 1:
            logger.info("On pulse residual is similar to left off pulse region.")

        logger.info("Running on pulse - off pulse (R) test")
        ofr_on = tests(off_pulse_right, on_pulse, ntest=2)
        if ofr_on == 1:
            logger.info("On pulse residual is similar to right off pulse region.")

        return np.any([ofl_on, ofr_on])

    @property
    def model(self):
        """
        Function to make the model.

        Returns:
            2D array of spectrogram model.

        """
        logger.info(f"Making model.")
        if "all" in self.sgram_params.keys():
            model = self.model_from_params(
                [0], *self.sgram_params["all"]["popt"]
            ).reshape(self.nf, self.nt)
        else:
            assert len(self.sgram_params) == self.ncomponents
            logger.info(f"Found {self.ncomponents} components.")

            model = np.zeros(shape=(self.nf, self.nt))
            for i in range(1, self.ncomponents + 1):
                popt = self.sgram_params[i]["popt"]
                self.sgram_model.forfit = False
                model += self.sgram_model.evaluate([0], *popt)
        return model

    def model_from_params(self, x, *params):
        """
        Function to make the model using spectrogram parameters.

        Returns:
            Flattened array of spectrogram model.

        """
        assert len(params) % len(self.param_names) == 0
        ncomp = int(len(params) / len(self.param_names))
        nparams = int(len(self.param_names))
        model = np.zeros(shape=(self.nf, self.nt))
        for i in range(1, ncomp + 1):
            popt = params[(i - 1) * nparams: i * nparams]
            self.sgram_model.forfit = False
            model += self.sgram_model.evaluate([0], *popt)
        return model.ravel()

    def get_physical_params(self, mapping, params=None, errors=None):
        """
        Convert parameters to physical units using the input mapping.

        Args:
            mapping: Mapping to convert the parameters to physical units.
            params: parameters to be converted.
            errors: errors to be converted.

        Returns:
            physical_dict: Dictionary of physical parameters.
            physical_errs: Dictionary with errors converted to physical units.

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
