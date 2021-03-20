#!/usr/bin/env python3

import logging

import numpy as np

from burstfit.curvefit import CurveFit
from burstfit.mcmc import MCMC
from burstfit.utils.math import tests, transform_parameters
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
        mask: RFI channel mask array
        mcmcfit: To run MCMC after curve_fit
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
        mask=np.array([False]),
        mcmcfit=False,
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
        self.mcmc_params = {}
        self.profile_bounds = {}
        self.spectra_bounds = {}
        self.sgram_bounds = {}
        self.physical_params = {}
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
        self.reduced_chi_sq = None
        self.mask = mask
        self.residual_std = None
        self.mcmcfit = mcmcfit
        self.mcmc = None
        self.off_pulse_ts_std = None

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
        )
        logger.debug("Setting sgram_model attributes.")
        self.sgram_model.metadata = self.metadata
        self.sgram_model.mask = self.mask
        self.sgram_model.forfit = True
        self.sgram_model.clip_fac = self.clip_fac

    def make_spectra(self):
        """
        Make the spectra by using the profile fitting parameters.

        Returns:

        """
        tau_width = 0
        try:
            logger.info("Making spectra using profile fit parameters.")
            mu_idx = np.where(np.array(self.profile_param_names) == "mu_t")[0]
            sig_idx = np.where(np.array(self.profile_param_names) == "sigma_t")[0]
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

    def fitcycle(
        self,
        plot=False,
        profile_bounds=[],
        spectra_bounds=[],
        sgram_bounds=[-np.inf, np.inf],
    ):
        """
        Run the fitting cycle to fit one component

        Args:
            profile_bounds: Bounds for initial profile fit
            spectra_bounds: Bounds for initial spectra fit
            plot: To plot
            sgram_bounds: Bounds for spectrogram fitting

        Returns:

        """
        logger.info(f"Fitting component {self.comp_num}.")
        self.validate()
        self.precalc()
        self.initial_profilefit(plot=plot, bounds=profile_bounds)
        self.make_spectra()
        self.initial_spectrafit(plot=plot, bounds=spectra_bounds)
        _ = self.sgram_params.pop("all", None)
        self.sgram_fit(plot=plot, bounds=sgram_bounds)
        self.reduced_chi_sq = self.calc_redchisq()

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
        cf = CurveFit(
            function=self.sgram_model.pulse_model.function,
            xdata=xdata,
            ydata=ydata,
            bounds=bounds,
            retry=False,
        )
        popt, err = cf.run_fit()
        self.profile_params[self.comp_num] = {"popt": list(popt), "perr": err}
        self.profile_bounds[self.comp_num] = bounds

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
        cf = CurveFit(
            function=self.sgram_model.spectra_model.function,
            xdata=xdata,
            ydata=ydata,
            bounds=bounds,
            retry=False,
        )
        popt, err = cf.run_fit()
        self.spectra_params[self.comp_num] = {"popt": list(popt), "perr": err}
        self.spectra_bounds[self.comp_num] = bounds

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
        self.sgram_model.forfit = True
        logger.info(f"initial estimate for parameters: {p0}")
        cf = CurveFit(
            function=self.sgram_model.evaluate,
            xdata=[0],
            ydata=self.residual.ravel(),
            p0=p0,
            bounds=bounds,
            retry=True,
            retry_frac_runtimeerror=0.9,
            retry_frac_infinite_err=0.2,
        )
        popt, err = cf.run_fit()
        self.sgram_params[self.comp_num] = {"popt": list(popt), "perr": err}
        self.sgram_bounds[self.comp_num] = bounds

        logger.info(f"Converged parameters are:")
        for i, p in enumerate(self.sgram_params[self.comp_num]["popt"]):
            logger.info(f"{self.param_names[i]}: {p} +- {err[i]}")

        if plot:
            plot_2d_fit(
                self.residual,
                self.sgram_model.evaluate,
                self.sgram_params[self.comp_num]["popt"],
                self.tsamp,
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
        self.sgram_model.forfit = True
        bounds = [-np.inf, np.inf]
        cf = CurveFit(
            function=self.model_from_params,
            xdata=[0],
            ydata=self.sgram.ravel(),
            p0=p0,
            bounds=bounds,
            retry=True,
            retry_frac_runtimeerror=0.9,
            retry_frac_infinite_err=0.2,
        )
        popt, err = cf.run_fit()
        self.sgram_params["all"] = {}
        for i in range(self.ncomponents):
            po = popt[i * self.sgram_model.nparams : (i + 1) * self.sgram_model.nparams]
            pe = err[i * self.sgram_model.nparams : (i + 1) * self.sgram_model.nparams]
            self.sgram_params["all"][i + 1] = {"popt": list(po), "perr": pe}
        self.sgram_bounds["all"] = bounds

        logger.info(f"Converged parameters are:")
        for i in range(1, self.ncomponents + 1):
            logger.info(f"Component {i}")
            params = self.sgram_params["all"][i]["popt"]
            errors = self.sgram_params["all"][i]["perr"]
            for j, p in enumerate(params):
                logger.info(f"{self.param_names[j]}: {p} +- {errors[j]}")

        if plot:
            plot_2d_fit(self.sgram, self.model_from_params, popt, self.tsamp)

        self.residual = self.sgram - self.model
        self.reduced_chi_sq = self.calc_redchisq()

    def fitall(
        self,
        plot=True,
        max_ncomp=5,
        profile_bounds=[],
        spectra_bounds=[],
        sgram_bounds=[-np.inf, np.inf],
        **mcmc_kwargs,
    ):
        """
        Perform spectro-temporal fitting on the spectrogram for all the components.

        Args:
            spectra_bounds: Bounds for initial profile fit
            profile_bounds: Bounds for initial spectra fit
            plot: to plot the fitting results.
            max_ncomp: maximum number of components to fit.
            sgram_bounds: bounds on spectrogram fit.
            **mcmc_kwargs: arguments for mcmc

        Returns:

        """
        self.precalc()
        test_res = self.run_tests
        if test_res:
            logger.warning(
                "On pulse region looks like noise. Check candidate parameters"
            )

        while self.ncomponents < max_ncomp:
            if np.any(profile_bounds):
                logger.warning(
                    f"Input profile bounds detected. Using them for component {self.comp_num}"
                )
            if np.any(spectra_bounds):
                logger.warning(
                    f"Input spectra bounds detected. Using them for component {self.comp_num}"
                )
            self.fitcycle(
                plot=plot,
                profile_bounds=profile_bounds,
                spectra_bounds=spectra_bounds,
                sgram_bounds=sgram_bounds,
            )
            test_res = self.run_tests
            if test_res:
                logger.info(
                    "On pulse residual looks like noise. "
                    "Terminating individual component fitting."
                )
                break
            self.comp_num += 1

        if self.comp_num > max_ncomp:
            logger.info(
                "Max number of components reached. "
                "Terminated individual component fitting."
            )
            self.comp_num -= 1

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
                    "On pulse residual does not look like noise."
                    "Check the fitting results carefully."
                )
        else:
            popt = self.sgram_params[1]["popt"].copy()
            err = self.sgram_params[1]["perr"].copy()
            self.sgram_params["all"] = {}
            self.sgram_params["all"][1] = {"popt": popt, "perr": err}
            logger.info(f"Final number of components = 1. Terminating fitting.")

        self.residual_std = np.std(self.residual.sum(0))

        if self.mcmcfit:
            self.run_mcmc(plot=plot, **mcmc_kwargs)

        self.off_pulse_ts_std = np.std(self.get_off_pulse_region().sum(0))

    def run_mcmc(
        self,
        plot=False,
        nwalkers=30,
        nsteps=1000,
        skip=3000,
        ncores=10,
        start_pos_dev=0.01,
        prior_range=0.5,
        save_results=True,
        outname=None,
    ):
        """
        Runs MCMC using the final fit parameters.

        Args:
            plot: To plot the outputs.
            nwalkers: Number of walkers for MCMC.
            nsteps: Number of iterations for MCMC.
            skip: Number of samples to skip to remove burn-in.
            ncores: Number of CPUs to use.
            start_pos_dev: Percent deviation for start position of the samples
            prior_range: Percent of initial guess to set as prior range
            save_results: Save MCMC samples to a file
            outname: Name of output files

        Returns:

        """
        if not outname:
            if self.outname:
                outname = self.outname
            else:
                outname = "mcmc_res"

        param_names = []
        for i in range(1, self.ncomponents + 1):
            for p in self.param_names:
                param_names += [p + "_" + str(i)]
        self.mcmc = MCMC(
            self.model_from_params,
            self.sgram,
            self.sgram_params["all"],
            param_names,
            nwalkers,
            nsteps,
            skip,
            start_pos_dev,
            prior_range,
            ncores,
            outname,
            save_results,
        )
        self.mcmc.run_mcmc()
        if not np.any(self.mcmc.samples):
            self.mcmc.get_chain()

        qs = np.quantile(self.mcmc.samples, [0.16, 0.5, 0.84], axis=0)
        e1 = qs[1] - qs[0]
        e2 = qs[2] - qs[1]
        err = [[e1[i], e2[i]] for i in range(len(e1))]
        popt = qs[1]
        for i in range(self.ncomponents):
            po = popt[i * self.sgram_model.nparams : (i + 1) * self.sgram_model.nparams]
            pe = err[i * self.sgram_model.nparams : (i + 1) * self.sgram_model.nparams]
            self.mcmc_params[i + 1] = {"popt": list(po), "perr": pe}
        self.mcmc.print_results()

        self.residual = self.sgram - self.model
        self.reduced_chi_sq = self.calc_redchisq()

        if plot:
            self.mcmc.plot(save=save_results)
            self.mcmc.make_autocorr_plot(save=save_results)
            qs = np.quantile(self.mcmc.samples, [0.5], axis=0)
            plot_2d_fit(self.sgram, self.model_from_params, qs[0], self.tsamp)
        return self.mcmc

    def get_off_pulse_region(self):
        """
        Returns off pulse region (2D) using fit parameters.

        Returns:

        """
        if "mu_t" and "sigma_t" in self.param_names:
            logger.info(
                "mu_t and sigma_t found in params. Using those to estimate off pulse region."
            )
            mu_idx = np.where(np.array(self.param_names) == "mu_t")[0][0]
            sig_idx = np.where(np.array(self.param_names) == "sigma_t")[0][0]

            if self.mcmc_params:
                logger.info("Using MCMC parameters")
                params = self.mcmc_params
            elif "all" in self.sgram_params.keys():
                logger.info("Using sgram all-component-fit parameters.")
                params = self.sgram_params["all"]
            else:
                logger.info("Using sgram fit parameters.")
                params = self.sgram_params

            mus = []
            sigs = []
            for k in params.keys():
                mus.append(params[k]["popt"][mu_idx])
                sigs.append(params[k]["popt"][sig_idx])

            min_mu = np.min(mus)
            max_sigma = np.max(sigs)
            end = int(min_mu - 3 * 2.355 * max_sigma)
        else:
            end = self.nt // 2 - 3 * self.width
        return self.sgram[:, :end].copy()

    @property
    def run_tests(self):
        """
        Run statistical tests to compare ON pulse residual with OFF pulse spectrogram distributions.

        Returns:
            True if either of the left or right OFF pulse regions are similar to the residual ON pulse region.

        """
        logger.info(f"Running statistical tests on the residual.")
        on_pulse = self.residual[:, self.i0 - self.width : self.i0 + self.width]
        off_pulse_left = self.sgram[:, 0 : 2 * self.width]
        off_pulse_right = self.sgram[:, -2 * self.width :]
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

    def calc_redchisq(self):
        """
        Calculates reduced chi-square value of the fit using sgram, model and off pulse standard deviation.

        Returns:
            Reduced chi-square value of the fit

        """
        logger.debug("Estimating reduced chi square value of the fit.")
        sg = self.get_off_pulse_region()
        std = np.std(sg)
        logger.debug(f"Standard deviation is {std}")
        chi_squared = np.sum(((self.sgram - self.model) / std) ** 2)
        data_size = (~self.sgram.mask).sum()
        logger.debug(f"Unmasked data size is {data_size}")
        reduced_chi_squared = chi_squared / (
            data_size - self.ncomponents * len(self.param_names)
        )
        logger.info(f"Reduced chi-square value of fit is: {reduced_chi_squared}")
        return reduced_chi_squared

    @property
    def model(self):
        """
        Function to make the model.

        Returns:
            2D array of spectrogram model.

        """
        logger.info(f"Making model.")
        if self.mcmc_params:
            dict = self.mcmc_params
        else:
            if "all" in self.sgram_params.keys():
                dict = self.sgram_params["all"]
            else:
                dict = self.sgram_params

        assert len(dict) == self.ncomponents
        logger.info(f"Found {self.ncomponents} components.")

        model = np.zeros(shape=(self.nf, self.nt))
        if self.sgram_model.forfit:
            model = model.ravel()
        for i in range(1, self.ncomponents + 1):
            popt = dict[i]["popt"]
            model += self.sgram_model.evaluate([0], *popt)
        if self.sgram_model.forfit and self.clip_fac != 0:
            model = np.clip(model, 0, self.clip_fac)
        return model.reshape((self.nf, self.nt))

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
        model = np.ma.masked_array(model)
        if self.sgram_model.forfit:
            model = model.ravel()
        for i in range(1, ncomp + 1):
            popt = params[(i - 1) * nparams : i * nparams]
            model += self.sgram_model.evaluate([0], *popt)
        if self.sgram_model.forfit and self.clip_fac != 0:
            model = np.clip(model, 0, self.clip_fac)
        return model

    def get_physical_parameters(self, my_mapping):
        """
        Function to use the my_mapping function and convert fitted parameters to physical units

        Args:
            my_mapping: function to map parameter dictionary to a mapping dictionary for parameters

        Returns:

        """
        for comp in self.sgram_params.keys():
            logger.info(f"Converting parameters of component: {comp}")
            if comp == "all":
                self.physical_params[comp] = {}
                for i in range(1, self.ncomponents + 1):
                    params = self.sgram_params[comp][i]
                    mapping = my_mapping(params, self)
                    self.physical_params[comp][i] = transform_parameters(
                        params, mapping, self.param_names
                    )
            else:
                params = self.sgram_params[comp]
                mapping = my_mapping(params, self)
                self.physical_params[comp] = transform_parameters(
                    params, mapping, self.param_names
                )
        return self.physical_params
