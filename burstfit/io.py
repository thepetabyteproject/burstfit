import json
from burstfit.utils.misc import MyEncoder
from burstfit.model import Model, SgramModel
from burstfit.utils.functions import pulse_fn, spectra_fn, sgram_fn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BurstIO:
    def __init__(self, BurstFit=None, BurstData=None, dictionary=None, jsonfile=None):
        self.burstfit = BurstFit
        self.burstdata = BurstData
        self.jsonfile = jsonfile
        self.dictionary = dictionary

    @property
    def set_attributes_to_save(self):
        self.fileheader = vars(self.burstdata.your_header)
        if not isinstance(self.fileheader["dtype"], str):
            self.fileheader["dtype"] = self.fileheader["dtype"].__name__
        self.nstart = self.burstdata.nstart
        self.tcand = self.burstdata.tcand
        self.mask = self.burstdata.mask

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
        return self

    def save_results(self, outname=None):
        self.set_attributes_to_save
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

        for k in keys_to_use:
            out[k] = getattr(self, k)
        if not outname:
            outname = self.outname + ".json"

        with open(outname, "w") as fp:
            json.dump(out, fp, cls=MyEncoder, indent=4)
        return out

    def read_json_and_precalc(self, file=None):
        if file:
            self.jsonfile = file
        if not self.jsonfile:
            raise AttributeError(f"self.jsonfile not set.")
        with open(self.jsonfile, "r") as fp:
            self.dictionary = json.load(fp)

        self.set_classes_from_dict()
        for k in self.dictionary.keys():
            setattr(self, k, self.dictionary[k])
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
        if self.dictionary["pulse_function"] == "pulse_fn":
            pulseModel = Model(pulse_fn)
        else:
            raise ValueError(f"self.dictionary['pulse_function'] not supported.")

        if self.dictionary["spectra_function"] == "spectra_fn":
            spectraModel = Model(spectra_fn)
        else:
            raise ValueError(f"self.dictionary['spectra_function'] not supported.")

        if self.dictionary["sgram_function"] == "sgram_fn":
            self.sgramModel = SgramModel(pulseModel, spectraModel, sgram_fn)
        else:
            raise ValueError(f"self.dictionary['sgram_function'] not supported.")

    def model(self):
        logging.info(f"Making model.")
        assert len(self.sgram_params) == self.ncomponents
        logging.info(f"Found {self.ncomponents} components.")

        model = np.zeros(shape=(self.nf, self.nt))
        for i in range(1, self.ncomponents + 1):
            popt = self.sgram_params[str(i)]["popt"]
            self.sgramModel.forfit = False
            model += self.sgramModel.evaluate([0], *popt)
        return model
