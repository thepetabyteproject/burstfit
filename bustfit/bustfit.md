<a name="burstfit.fit"></a>
# burstfit.fit

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L3)

<a name="burstfit.fit.BurstFit"></a>
## BurstFit Objects

```python
class BurstFit()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L15)

BurstFit class to perform spectro-temporal modeling on the burst data

**Arguments**:

- `sgram_model` - Spectrogram Model class object
- `sgram` - 2D array of spectrogram
- `width` - width of the candidate
- `dm` - DM of the candidate
- `foff` - frequency resolution of the data
- `fch1` - Frequency of first channel (MHz))
- `tsamp` - Sampling interval (seconds)
- `clip_fac` - Clip factor based on nbits of data
- `outname` - Outname for the outputs
- `mask` - RFI channel mask array
- `mcmcfit` - To run MCMC after curve_fit

<a name="burstfit.fit.BurstFit.ncomponents"></a>
#### ncomponents

```python
 | @property
 | def ncomponents()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L83)

Returns: number of components

<a name="burstfit.fit.BurstFit.validate"></a>
#### validate

```python
 | def validate()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L96)

Validate the class attributes

**Returns**:


<a name="burstfit.fit.BurstFit.precalc"></a>
#### precalc

```python
 | def precalc()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L111)

Perform precalculations for fitting

**Returns**:


<a name="burstfit.fit.BurstFit.make_spectra"></a>
#### make\_spectra

```python
 | def make_spectra()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L145)

Make the spectra by using the profile fitting parameters.

**Returns**:


<a name="burstfit.fit.BurstFit.fitcycle"></a>
#### fitcycle

```python
 | def fitcycle(plot=False, profile_bounds=[], spectra_bounds=[], sgram_bounds=[-np.inf, np.inf])
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L198)

Run the fitting cycle to fit one component

**Arguments**:

- `profile_bounds` - Bounds for initial profile fit
- `spectra_bounds` - Bounds for initial spectra fit
- `plot` - To plot
- `sgram_bounds` - Bounds for spectrogram fitting
  

**Returns**:


<a name="burstfit.fit.BurstFit.initial_profilefit"></a>
#### initial\_profilefit

```python
 | def initial_profilefit(plot=False, bounds=[])
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L227)

Perform initial profile fit on the pulse.

**Arguments**:

- `plot` - To plot the fit result.
- `bounds` - Bounds for fitting.
  

**Returns**:


<a name="burstfit.fit.BurstFit.initial_spectrafit"></a>
#### initial\_spectrafit

```python
 | def initial_spectrafit(plot=False, bounds=[])
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L284)

Perform initial spectra fit on the spectra.

**Arguments**:

- `plot` - To plot the fitting results.
- `bounds` - Bounds for fitting.
  

**Returns**:


<a name="burstfit.fit.BurstFit.sgram_fit"></a>
#### sgram\_fit

```python
 | def sgram_fit(plot=False, bounds=[-np.inf, np.inf])
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L331)

Perform fit on the spectrogram and updates the residual.

**Arguments**:

- `plot` - To plot the fitting results.
- `bounds` - Bounds on the spectrogram fit.
  

**Returns**:


<a name="burstfit.fit.BurstFit.fit_all_components"></a>
#### fit\_all\_components

```python
 | def fit_all_components(plot)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L378)

Fit all components together (used if num_comp > 1)

**Arguments**:

- `plot` - To plot the fitting results.
  

**Returns**:


<a name="burstfit.fit.BurstFit.fitall"></a>
#### fitall

```python
 | def fitall(plot=True, max_ncomp=5, profile_bounds=[], spectra_bounds=[], sgram_bounds=[-np.inf, np.inf], **mcmc_kwargs, ,)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L426)

Perform spectro-temporal fitting on the spectrogram for all the components.

**Arguments**:

- `spectra_bounds` - Bounds for initial profile fit
- `profile_bounds` - Bounds for initial spectra fit
- `plot` - to plot the fitting results.
- `max_ncomp` - maximum number of components to fit.
- `sgram_bounds` - bounds on spectrogram fit.
- `**mcmc_kwargs` - arguments for mcmc
  

**Returns**:


<a name="burstfit.fit.BurstFit.run_mcmc"></a>
#### run\_mcmc

```python
 | def run_mcmc(plot=False, nwalkers=30, nsteps=1000, skip=3000, ncores=10, start_pos_dev=0.01, prior_range=0.5, save_results=True, outname=None)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L515)

Runs MCMC using the final fit parameters.

**Arguments**:

- `plot` - To plot the outputs.
- `nwalkers` - Number of walkers for MCMC.
- `nsteps` - Number of iterations for MCMC.
- `skip` - Number of samples to skip to remove burn-in.
- `ncores` - Number of CPUs to use.
- `start_pos_dev` - Percent deviation for start position of the samples
- `prior_range` - Percent of initial guess to set as prior range
- `save_results` - Save MCMC samples to a file
- `outname` - Name of output files
  

**Returns**:


<a name="burstfit.fit.BurstFit.get_off_pulse_region"></a>
#### get\_off\_pulse\_region

```python
 | def get_off_pulse_region()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L593)

Returns off pulse region (2D) using fit parameters.

**Returns**:


<a name="burstfit.fit.BurstFit.run_tests"></a>
#### run\_tests

```python
 | @property
 | def run_tests()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L631)

Run statistical tests to compare ON pulse residual with OFF pulse spectrogram distributions.

**Returns**:

  True if either of the left or right OFF pulse regions are similar to the residual ON pulse region.

<a name="burstfit.fit.BurstFit.calc_redchisq"></a>
#### calc\_redchisq

```python
 | def calc_redchisq()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L661)

Calculates reduced chi-square value of the fit using sgram, model and off pulse standard deviation.

**Returns**:

  Reduced chi-square value of the fit

<a name="burstfit.fit.BurstFit.model"></a>
#### model

```python
 | @property
 | def model()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L683)

Function to make the model.

**Returns**:

  2D array of spectrogram model.

<a name="burstfit.fit.BurstFit.model_from_params"></a>
#### model\_from\_params

```python
 | def model_from_params(x, *params)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L713)

Function to make the model using spectrogram parameters.

**Returns**:

  Flattened array of spectrogram model.

<a name="burstfit.fit.BurstFit.get_physical_parameters"></a>
#### get\_physical\_parameters

```python
 | def get_physical_parameters(my_mapping)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/fit.py#L735)

Function to use the my_mapping function and convert fitted parameters to physical units

**Arguments**:

- `my_mapping` - function to map parameter dictionary to a mapping dictionary for parameters
  

**Returns**:


