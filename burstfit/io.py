import json
import logging

import numpy as np

from burstfit.model import Model, SgramModel
from burstfit.utils.functions import pulse_fn, gauss_norm, gauss_norm2, gauss_norm3, sgram_fn
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
    """

    def __init__(
            self, burstfit_obj=None, burstdata_obj=None, dictionary=None, jsonfile=None
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
        self.sgramModel = None
        self.pulse_function = None
        self.spectra_function = None
        self.ncomponents = None
        self.tsamp = None
        self.outname = None

    def set_attributes_to_save(self):
        """
        Sets required attributes to be saved

        Returns:

        """
        logger.info("Setting attributes to be saved.")
        logger.info("Reading attributes from BurstData object.")
        self.fileheader = vars(self.burstdata.your_header)
        if not isinstance(self.fileheader["dtype"], str):
            self.fileheader["dtype"] = self.fileheader["dtype"].__name__
        self.nstart = self.burstdata.nstart
        self.tcand = self.burstdata.tcand
        self.mask = self.burstdata.mask
        self.id = self.burstdata.id

        logger.info("Reading attributes from BurstFit object.")
        for k in self.burstfit.__dict__.keys():
            if ("sgram" == k) or ("residual" in k) or ("ts" in k) or ("spectra" == k):
                continue
            elif "model" in k:
                sgram_model_class = getattr(self.burstfit, k)
                self.sgramModel = sgram_model_class
                self.sgram_function = self.sgramModel.sgram_function.__name__
                self.pulse_function = self.sgramModel.pulse_model.function.__name__
                self.spectra_function = self.sgramModel.spectra_model.function.__name__
            else:
                setattr(self, k, getattr(self.burstfit, k))
        self.ncomponents = self.burstfit.ncomponents
        self.tsamp = self.burstfit.tsamp
        logger.info("Copied necessary attributes")
        return self

    def save_results(self, outname=None):
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
            "sgramModel",
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

        logger.info(f"Writing JSON file: {outname}.")
        with open(outname, "w") as fp:
            json.dump(out, fp, cls=MyEncoder, indent=4)
        return out

    def read_json_and_precalc(self, file=None):
        """
        Read the result json file and calculate required parameters.

        Args:
            file: results file to read

        Returns:

        """
        if file:
            self.jsonfile = file
        if not self.jsonfile:
            raise AttributeError(f"self.jsonfile not set.")
        logger.info(f"Reading JSON: {self.jsonfile}")
        with open(self.jsonfile, "r") as fp:
            self.dictionary = json.load(fp)

        logger.info(f"Setting models (pulse, spectra and spectrogram).")
        self.set_classes_from_dict()

        logger.info(f"Setting I/O class attributes.")
        for k in self.dictionary.keys():
            setattr(self, k, self.dictionary[k])

        self.set_metadata()
        logger.info(f"BurstIO class is ready with necessary attributes.")

    def set_metadata(self):
        """
        Sets the metadata tuple

        Returns:

        """
        self.sgramModel.metadata = (
            self.nt,
            self.nf,
            self.dm,
            self.tsamp,
            self.fch1,
            self.foff,
            self.clip_fac,
        )

    def set_classes_from_dict(self):
        """
        Sets models and required classes

        Returns:

        """
        if self.dictionary["pulse_function"] == "pulse_fn":
            pulseModel = Model(pulse_fn)
        else:
            raise ValueError(f"{self.dictionary['pulse_function']} not supported.")

        if self.dictionary["spectra_function"] == "gauss_norm":
            spectraModel = Model(gauss_norm)
        elif self.dictionary["spectra_function"] == "gauss_norm2":
            spectraModel = Model(gauss_norm2)
        elif self.dictionary["spectra_function"] == "gauss_norm3":
            spectraModel = Model(gauss_norm3)
        else:
            raise ValueError(f"{self.dictionary['spectra_function']} not supported.")

        if self.dictionary["sgram_function"] == "sgram_fn":
            self.sgramModel = SgramModel(pulseModel, spectraModel, sgram_fn)
        else:
            raise ValueError(f"{self.dictionary['sgram_function']} not supported.")

    @property
    def model(self):
        """
        Function to make the model

        Returns:
            2D array of model

        """
        logging.info(f"Making model.")
        assert len(self.sgram_params) == self.ncomponents
        logging.info(f"Found {self.ncomponents} components.")

        model = np.zeros(shape=(self.nf, self.nt))
        for i in range(1, self.ncomponents + 1):
            popt = self.sgram_params[str(i)]["popt"]
            self.sgramModel.forfit = False
            model += self.sgramModel.evaluate([0], *popt)
        return model
