import logging

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
        if not param_names:
            c = self.function.__code__
            param_names = list(c.co_varnames[1 : c.co_argcount])
        self.param_names = param_names
        self.params = params

    def evaluate(self, x, *params):
        """
        Evaluate the function at input parameters

        Args:
            x:
            *params:

        Returns:

        """
        return self.function(x, *params)

    def get_param_dict(self, *params):
        """
        Convert input param list to a dictionary with param_names

        Args:
            *params:

        Returns:

        """
        f = {}
        for i, p in enumerate(params):
            f[self.param_names[i]] = p
        self.params = f
        return self.params

    @property
    def nparams(self):
        """

        Returns: number of parameters

        """
        return len(self.param_names)


class SgramModel:
    """
    Wrapper class for spectrogram model

    Args:
        pulse_model: Model instance of pulse function
        spectra_model: Model instance of spectra function
        sgram_fn: Spectrogram function
        metadata: Metadata for sgram function
        param_names: names of sgram parameters

    """

    def __init__(
        self,
        pulse_model=None,
        spectra_model=None,
        sgram_fn=None,
        metadata=None,
        param_names=None,
    ):
        self.pulse_model = pulse_model
        self.spectra_model = spectra_model
        self.sgram_function = sgram_fn
        self.metadata = metadata
        self.forfit = True
        if not param_names:
            self.param_names = (
                self.spectra_model.param_names + self.pulse_model.param_names + ["DM"]#, "tau_idx"]
            )
        else:
            self.param_names = param_names
            
    @property
    def nparams(self):
        """

        Returns: number of parameters

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
        np = self.pulse_model.nparams
        assert len(params) == len(self.param_names)
        spectra_params = self.spectra_model.get_param_dict(*params[0:ns])
        pulse_params = self.pulse_model.get_param_dict(*params[ns : ns + np])
        other_params = params[ns + np:]
        model = self.sgram_function(
            self.metadata,
            self.pulse_model.function,
            self.spectra_model.function,
            spectra_params,
            pulse_params,
            other_params,
        )
        if self.forfit:
            return model.ravel()
        else:
            return model
