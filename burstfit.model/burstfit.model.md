<a name="burstfit.model"></a>
# burstfit.model

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/779f2e5a427208ec6a7c1b8fc49d361116c50dcc/burstfit/model.py#L1)

<a name="burstfit.model.Model"></a>
## Model Objects

```python
class Model()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/779f2e5a427208ec6a7c1b8fc49d361116c50dcc/burstfit/model.py#L8)

Wrapper class for spectra and profile functions.

**Arguments**:

- `function` - Fitting function
- `param_names` - List of parameter names
- `params` - list of parameter values

<a name="burstfit.model.Model.evaluate"></a>
#### evaluate

```python
 | def evaluate(x, *params)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/779f2e5a427208ec6a7c1b8fc49d361116c50dcc/burstfit/model.py#L33)

Evaluate the function at input parameters

**Arguments**:

- `x` - input x values to evaluate function
- `*params` - params to use in the function
  

**Returns**:


<a name="burstfit.model.Model.get_param_dict"></a>
#### get\_param\_dict

```python
 | def get_param_dict(*params, *, keys="params")
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/779f2e5a427208ec6a7c1b8fc49d361116c50dcc/burstfit/model.py#L46)

Convert input param list to a dictionary with param_names

**Arguments**:

- `*params` - parameter list
  

**Returns**:


<a name="burstfit.model.Model.nparams"></a>
#### nparams

```python
 | @property
 | def nparams()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/779f2e5a427208ec6a7c1b8fc49d361116c50dcc/burstfit/model.py#L71)

**Returns**:

  number of parameters

<a name="burstfit.model.SgramModel"></a>
## SgramModel Objects

```python
class SgramModel()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/779f2e5a427208ec6a7c1b8fc49d361116c50dcc/burstfit/model.py#L81)

Wrapper class for spectrogram model

**Arguments**:

- `pulse_model` - Model instance of pulse function
- `spectra_model` - Model instance of spectra function
- `sgram_fn` - Spectrogram function
- `metadata` - Metadata for sgram function
- `param_names` - names of sgram parameters
- `clip_fac` - clipping factor

<a name="burstfit.model.SgramModel.nparams"></a>
#### nparams

```python
 | @property
 | def nparams()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/779f2e5a427208ec6a7c1b8fc49d361116c50dcc/burstfit/model.py#L122)

**Returns**:

  number of parameters

<a name="burstfit.model.SgramModel.evaluate"></a>
#### evaluate

```python
 | def evaluate(x, *params)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/779f2e5a427208ec6a7c1b8fc49d361116c50dcc/burstfit/model.py#L131)

Function to evaluate sgram_function at input parameters

**Arguments**:

- `x` - Dummy input. Not used.
- `*params` - Parameters to evaluate sgram_function at.
  

**Returns**:

- `model` - 2D array of model

