import json
import logging
import os

import numpy as np

from burstfit.model import Model, SgramModel
from burstfit.utils.functions import (
    pulse_fn,
    gauss,
    gauss_norm,
    gauss_norm2,
    gauss_norm3,
    sgram_fn,
    pulse_fn_vec,
    sgram_fn_vec,
)
from burstfit.utils.misc import MyEncoder

logger = logging.getLogger(__name__)


class BurstIO:
    """
    I/O class to save the fitting results and read results to reproduce model.

    Args:
        burstfit_obj: Instance of burstfit class with fitting parameters
        burstdata_obj: Instance of burstdata class with burst data
        dictionary: dictionary with fitting results
        jsonfile: JSON file with the fitting results
        outname:
        outdir:
    """

    def __init__(
        self,
        burstfit_obj=None,
        burstdata_obj=None,
        dictionary=None,
        jsonfile=None,
        outname=None,
        outdir=None,
    ):
        self.burstfit = burstfit_obj
        self.burstdata = burstdata_obj
        self.jsonfile = jsonfile
        self.dictionary = dictionary
        self.fileheader = None
        self.nstart = None
        self.tcand = None
        self.mask = None
        self.id = None
        self.sgram_function = None
        self.sgram_model = None
        self.pulse_function = None
        self.spectra_function = None
        self.ncomponents = None
        self.tsamp = None
        self.outname = outname
        self.outdir = outdir
        self.mcmc = None

        if self.jsonfile:
            if not os.path.isfile(self.jsonfile):
                raise IOError(f"{self.jsonfile} not found")

    def set_attributes_to_save(self):
        """
        Sets required attributes to be saved

        Returns:

        """
        logger.info("Setting attributes to be saved.")
        if self.burstdata:
            logger.info("Reading attributes from BurstData object.")
            self.fileheader = vars(self.burstdata.your_header)
            if not isinstance(self.fileheader["dtype"], str):
                self.fileheader["dtype"] = self.fileheader["dtype"].__name__
            self.nstart = self.burstdata.nstart
            self.tcand = self.burstdata.tcand
            self.mask = self.burstdata.mask
            self.id = self.burstdata.id
        else:
            logger.warning(
                "BurstData object not found. Make sure that the values of following parameters have been "
                "set already: fileheader, nstart, tcand, mask, id."
            )

        logger.info("Reading attributes from BurstFit object.")
        for k in self.burstfit.__dict__.keys():
            if ("sgram" == k) or ("residual" == k) or ("ts" == k) or ("spectra" == k):
                continue
            elif "model" in k:
                sgram_model_class = getattr(self.burstfit, k)
                self.sgram_model = sgram_model_class
                self.sgram_function = self.sgram_model.sgram_function.__name__
                self.pulse_function = self.sgram_model.pulse_model.function.__name__
                self.spectra_function = self.sgram_model.spectra_model.function.__name__
            elif k == "mcmc":
                mcmc_class = getattr(self.burstfit, k)
                if mcmc_class:
                    remove = [
                        "sgram",
                        "sampler",
                        "samples",
                        "autocorr",
                        "pos",
                        "model_function",
                    ]
                    mcmc_dict = vars(mcmc_class).copy()
                    for r in remove:
                        mcmc_dict.pop(r, None)
                    self.mcmc = mcmc_dict
            else:
                setattr(self, k, getattr(self.burstfit, k))
        self.ncomponents = self.burstfit.ncomponents
        self.tsamp = self.burstfit.tsamp
        logger.info("Copied necessary attributes")
        return self

    def save_results(self, outname=None, outdir=None):
        """

        Saves results of parameter fitting

        Args:
            outname: name of the output json file

        Returns:

        """
        self.set_attributes_to_save()
        out = {}
        nots = [
            "burstfit",
            "burstdata",
            "metadata",
            "sgram_model",
            "jsonfile",
            "dictionary",
        ]
        keys = self.__dict__.keys()
        keys_to_use = []
        for k in keys:
            if k not in nots:
                keys_to_use.append(k)

        logger.info(f"Preparing dictionary to be saved.")
        for k in keys_to_use:
            out[k] = getattr(self, k)
        if not outname:
            if self.outname:
                outname = self.outname + ".json"
            else:
                outname = self.id + ".json"
        if not outdir:
            if self.outdir:
                outdir = self.outdir
            else:
                outdir = os.getcwd()
        logger.info(f"Writing JSON file: {outdir}/{outname}.")
        with open(outdir + "/" + outname, "w") as fp:
            json.dump(out, fp, cls=MyEncoder, indent=4)
        return out

    def read_json_and_precalc(
        self, file=None, pulse_function=None, spectra_function=None, sgram_function=None
    ):
        """
        Read the result json file and calculate required parameters.

        Args:
            file: results file to read
            pulse_function: Pulse function used for modeling
            spectra_function: Spectra function used for modeling
            sgram_function: Sgram function used for modeling

        Returns:

        """
        if file:
            self.jsonfile = file
        if not self.jsonfile:
            raise AttributeError(f"self.jsonfile not set.")
        logger.info(f"Reading JSON: {self.jsonfile}")
        with open(self.jsonfile, "r") as fp:
            self.dictionary = json.load(fp)

        logger.info(f"Setting I/O class attributes.")
        for k in self.dictionary.keys():
            setattr(self, k, self.dictionary[k])

        logger.info(f"Setting models (pulse, spectra and spectrogram).")
        self.set_classes_from_dict(
            pulse_function=pulse_function,
            spectra_function=spectra_function,
            sgram_function=sgram_function,
        )

        self.set_metadata()
        logger.info(f"BurstIO class is ready with necessary attributes.")

    def set_metadata(self):
        """
        Sets the metadata tuple

        Returns:

        """
        self.sgram_model.metadata = (
            self.nt,
            self.nf,
            self.dm,
            self.tsamp,
            self.fch1,
            self.foff,
        )

    def set_classes_from_dict(
        self, pulse_function=None, spectra_function=None, sgram_function=None
    ):
        """
        Sets models and required classes

        Args:
            pulse_function: Pulse function used for modeling
            spectra_function: Spectra function used for modeling
            sgram_function: Sgram function used for modeling

        Returns:

        """

        if pulse_function:
            assert self.dictionary["pulse_function"] == pulse_function.__name__, (
                f"pulse_function ({pulse_function.__name__}) should have the same name as that in "
                f"the JSON file ({self.dictionary['pulse_function']})"
            )
            pulseModel = Model(pulse_function)
        else:
            if self.dictionary["pulse_function"] == "pulse_fn":
                pulseModel = Model(pulse_fn)
            elif self.dictionary["pulse_function"] == "gauss":
                pulseModel = Model(gauss)
            elif self.dictionary["pulse_function"] == "pulse_fn_vec":
                pulseModel = Model(pulse_fn_vec)
            else:
                raise ValueError(
                    f"Function: {self.dictionary['pulse_function']} not supported. If it is not one of the functions "
                    f"available in BurstFit, then provide it as input (pulse_function)."
                )

        if spectra_function:
            assert self.dictionary["spectra_function"] == spectra_function.__name__, (
                f"spectra_function ({spectra_function.__name__}) should have the same name as that in "
                f"the JSON file ({self.dictionary['spectra_function']})"
            )
            spectraModel = Model(spectra_function)
        else:
            if self.dictionary["spectra_function"] == "gauss_norm":
                spectraModel = Model(gauss_norm)
            elif self.dictionary["spectra_function"] == "gauss_norm2":
                spectraModel = Model(gauss_norm2)
            elif self.dictionary["spectra_function"] == "gauss_norm3":
                spectraModel = Model(gauss_norm3)
            else:
                raise ValueError(
                    f"Function: {self.dictionary['spectra_function']} not supported. If it is not one of the functions "
                    f"available in BurstFit, then provide it as input (spectra_function)."
                )

        if sgram_function:
            assert self.dictionary["sgram_function"] == sgram_function.__name__, (
                f"sgram_function ({sgram_function.__name__}) should have the same name as that in "
                f"the JSON file ({self.dictionary['sgram_function']})"
            )
        else:
            if self.dictionary["sgram_function"] == "sgram_fn":
                sgram_function = sgram_fn
            elif self.dictionary["sgram_function"] == "sgram_fn_vec":
                sgram_function = sgram_fn_vec
            else:
                raise ValueError(
                    f"Function: {self.dictionary['sgram_function']} not supported. If it is not one of the functions "
                    f"available in BurstFit, then provide it as input (sgram_function)."
                )

        self.sgram_model = SgramModel(
            pulse_model=pulseModel,
            spectra_model=spectraModel,
            sgram_fn=sgram_function,
            clip_fac=self.clip_fac,
        )

    @property
    def model(self):
        """
        Function to make the model

        Returns:
            2D array of model

        """
        logging.info(f"Making model.")
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
        for key, value in dict.items():
            popt = value["popt"]
            model += self.sgram_model.evaluate([0], *popt)
        if self.sgram_model.forfit:
            model = np.clip(model, 0, self.clip_fac)
        return model.reshape((self.nf, self.nt))
