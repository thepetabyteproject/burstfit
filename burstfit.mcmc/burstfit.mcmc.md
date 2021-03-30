<a name="burstfit.mcmc"></a>
# burstfit.mcmc

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L3)

<a name="burstfit.mcmc.MCMC"></a>
## MCMC Objects

```python
class MCMC()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L15)

Class to run MCMC on the burst model.

**Arguments**:

- `model_function` - Function to create the model
- `sgram` - 2D spectrogram data
- `initial_guess` - Initial guess of parameters for MCMC (can be a dictionary or list)
- `param_names` - Names of parameters
- `nwalkers` - Number of walkers to use in MCMC
- `nsteps` - Number of iterations to use in MCMC
- `skip` - Number of samples to skip for burn-in
- `start_pos_dev` - Percent deviation for start position of the samples
- `prior_range` - Percent of initial guess to set as prior range
- `ncores` - Number of CPUs to use
- `outname` - Name of output files
- `save_results` - Save MCMC samples to a file

<a name="burstfit.mcmc.MCMC.ndim"></a>
#### ndim

```python
 | @property
 | def ndim()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L87)

Returns the number of dimensions.

**Returns**:

  number of dimensions

<a name="burstfit.mcmc.MCMC.lnprior"></a>
#### lnprior

```python
 | def lnprior(params)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L97)

Prior function. Priors are uniform from (1-prior_range)*initial_guess to (1+prior_range)*initial_guess.
Minimum prior for tau is set to 0.

**Arguments**:

- `params` - Parameters to check.
  

**Returns**:


<a name="burstfit.mcmc.MCMC.lnprob"></a>
#### lnprob

```python
 | def lnprob(params)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L116)

Log probability function.

**Arguments**:

- `params` - Parameters to evaluate at.
  

**Returns**:

  Prior + log likelihood at the inputs.

<a name="burstfit.mcmc.MCMC.lnlk"></a>
#### lnlk

```python
 | def lnlk(inps)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L132)

Log likelihood function. Uses the model_function to generate the model.

**Arguments**:

- `inps` - Parameters to evaluate at.
  

**Returns**:

  Log likelihood.

<a name="burstfit.mcmc.MCMC.set_initial_pos"></a>
#### set\_initial\_pos

```python
 | def set_initial_pos()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L146)

Function to set the initial values of walkers and prior ranges.
Minimum prior for tau is set to 0.

**Returns**:


<a name="burstfit.mcmc.MCMC.set_priors"></a>
#### set\_priors

```python
 | def set_priors()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L173)

Set priors for MCMC

**Returns**:


<a name="burstfit.mcmc.MCMC.run_mcmc"></a>
#### run\_mcmc

```python
 | def run_mcmc()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L243)

Runs the MCMC.

**Returns**:

  Sampler object

<a name="burstfit.mcmc.MCMC.get_chain"></a>
#### get\_chain

```python
 | def get_chain(skip=None)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L305)

Returns the chanins from sampler object after removing some samples for burn-in.

**Arguments**:

- `skip` - Number of steps to skip for burn-in.
  

**Returns**:

  Sample chain.

<a name="burstfit.mcmc.MCMC.print_results"></a>
#### print\_results

```python
 | def print_results()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L340)

Prints the results of MCMC analysis. It uses median values with 1-sigma errors based on MCMC posteriors.

**Returns**:


<a name="burstfit.mcmc.MCMC.plot"></a>
#### plot

```python
 | def plot(save=False)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L357)

Plot the samples and corner plot of MCMC posteriors.

**Arguments**:

- `save` - To save the corner plot.
  

**Returns**:


<a name="burstfit.mcmc.MCMC.make_autocorr_plot"></a>
#### make\_autocorr\_plot

```python
 | def make_autocorr_plot(save=False)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/mcmc.py#L372)

Make autocorrelation plot for MCMC (i.e autocorrelation  time scale vs iteration)
see https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

**Arguments**:

- `save` - To save the plot
  

**Returns**:


