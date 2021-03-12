import logging

import numpy as np

logger = logging.getLogger(__name__)


class Model:
    """
    Wrapper class for spectra and profile functions.

    Args:
        function: Fitting function
        param_names: List of parameter names
        params: list of parameter values


    """

    def __init__(self, function, param_names=None, params=None):
        self.function = function
        c = self.function.__code__
        self.function_input_names = list(c.co_varnames[1 : c.co_argcount])
        if not param_names:
            self.param_names = self.function_input_names
        else:
            assert len(param_names) == len(
                self.function_input_names
            ), "Length of param_names is not equal to number of input parameters of the function."
            self.param_names = param_names
        self.params = params

    def evaluate(self, x, *params):
        """
        Evaluate the function at input parameters

        Args:
            x: input x values to evaluate function
            *params: params to use in the function

        Returns:

        """
        return self.function(x, *params)

    def get_param_dict(self, *params, keys="params"):
        """
        Convert input param list to a dictionary with param_names

        Args:
            *params: parameter list

        Returns:

        """
        f = {}
        for i, p in enumerate(params):
            if keys == "params":
                f[self.param_names[i]] = p
            elif keys == "function":
                f[self.function_input_names[i]] = p
            else:
                raise ValueError(
                    "Invalid keys value. keys can only be params or function."
                )
        # self.params = f
        # return self.params
        return f

    @property
    def nparams(self):
        """

        Returns:
            number of parameters

        """
        return len(self.function_input_names)


class SgramModel:
    """
    Wrapper class for spectrogram model

    Args:
        pulse_model: Model instance of pulse function
        spectra_model: Model instance of spectra function
        sgram_fn: Spectrogram function
        metadata: Metadata for sgram function
        param_names: names of sgram parameters
        clip_fac: clipping factor

    """

    def __init__(
        self,
        pulse_model=None,
        spectra_model=None,
        sgram_fn=None,
        metadata=None,
        param_names=None,
        mask=np.array([False]),
        clip_fac=None,
    ):
        self.pulse_model = pulse_model
        self.spectra_model = spectra_model
        self.sgram_function = sgram_fn
        self.metadata = metadata
        self.forfit = True
        if not param_names:
            self.param_names = (
                self.spectra_model.param_names
                + self.pulse_model.param_names
                + ["DM"]  # , "tau_idx"]
            )
        else:
            self.param_names = param_names
        self.mask = mask
        self.clip_fac = clip_fac

    @property
    def nparams(self):
        """

        Returns:
            number of parameters

        """
        return len(self.param_names)

    def evaluate(self, x, *params):
        """
        Function to evaluate sgram_function at input parameters

        Args:
            x: Dummy input. Not used.
            *params: Parameters to evaluate sgram_function at.

        Returns:
            model: 2D array of model

        """
        ns = self.spectra_model.nparams
        nparam = self.pulse_model.nparams
        assert len(params) == len(self.param_names)
        spectra_params = self.spectra_model.get_param_dict(
            *params[0:ns], keys="function"
        )
        pulse_params = self.pulse_model.get_param_dict(
            *params[ns : ns + nparam], keys="function"
        )
        other_params = params[ns + nparam :]
        model = self.sgram_function(
            self.metadata,
            self.pulse_model.function,
            self.spectra_model.function,
            spectra_params,
            pulse_params,
            other_params,
        )
        model = np.ma.masked_array(model)
        if self.forfit:
            if self.clip_fac != 0:
                model = np.clip(model, 0, self.clip_fac)
            if self.mask.any():
                model[self.mask, :] = np.ma.masked
            return model.ravel()
        else:
            return model
