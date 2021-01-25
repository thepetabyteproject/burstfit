import numpy as np
import logging

logger = logging.getLogger(__name__)

class Model:
    def __init__(self, function, param_names=None, params=None):
        self.function = function
        if not param_names:
            c = self.function.__code__
            param_names = list(c.co_varnames[1:c.co_argcount])
        self.param_names = param_names
        self.params = params
        
    def evaluate(self, x, *params):
        return self.function(x, *params)
    
    def get_param_dict(self, *params):
        f = {}
        for i, p in enumerate(params):
            f[self.param_names[i]] = p
        self.params = f
        return self.params
    
    @property
    def nparams(self):
        return len(self.param_names)
    
    def get_physical_params(self, mapping, params=None, errors=None):
        if params:
            self.params = params
        if errors:
            self.errors = errors
            
        physical_dict = {}
        for key in mapping:
            physical_dict[key] = fma(*mapping[key])
        return physical_dict
                
    def fma(self, key, m, a):
        if self.errors:
            return self.params[key]*m + a, self.errors[key]*m + a 
        else:
            return self.params[key]*m + a
        

class SgramModel:
    def __init__(self, pulse_model=None, spectra_model=None, sgram_fn=None, 
                 metadata=None, param_names=None):
        self.pulse_model = pulse_model
        self.spectra_model = spectra_model
        self.sgram_function = sgram_fn
        self.metadata = metadata
        self.forfit = True
        if not param_names:
            self.param_names = self.spectra_model.param_names + self.pulse_model.param_names + ['DM']
        else:
            self.param_names = param_names
    
    def evaluate(self, x, *params):
        ns = self.spectra_model.nparams
        np = self.pulse_model.nparams
        assert len(params) == len(self.param_names)
        spectra_params = self.spectra_model.get_param_dict(*params[0: ns])
        pulse_params = self.pulse_model.get_param_dict(*params[ns: ns + np])
        dm = params[ns + np]
        model = self.sgram_function(self.metadata, self.pulse_model.function, 
                                   self.spectra_model.function, 
                                   spectra_params, pulse_params, dm)
        if self.forfit:
            return model.ravel()
        else:
            return model