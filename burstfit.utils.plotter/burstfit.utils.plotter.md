<a name="burstfit.utils.plotter"></a>
# burstfit.utils.plotter

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/utils/plotter.py#L1)

<a name="burstfit.utils.plotter.plot_1d_fit"></a>
#### plot\_1d\_fit

```python
def plot_1d_fit(xdata, ydata, function, popt, xlabel=None, ylabel=None, title=None, param_names=[], show=True, save=False, outname="1d_fit_res")
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/utils/plotter.py#L12)

Plot the results of 1D fits

**Arguments**:

- `xdata` - x value array
- `ydata` - original data values
- `function` - function used for fitting
- `popt` - fit parameters of the function
- `xlabel` - label of x axis
- `ylabel` - label of y axis
- `title` - title of the plot
- `param_names` - names of the parameters
  

**Returns**:


<a name="burstfit.utils.plotter.plot_2d_fit"></a>
#### plot\_2d\_fit

```python
def plot_2d_fit(sgram, function, popt, tsamp, title=None, show=True, save=False, outname="2d_fit_res", outdir=None)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/utils/plotter.py#L73)

Plot the result of spectrogram fit

**Arguments**:

- `sgram` - input 2D array of spectrogram
- `function` - spectrogram function used for fitting
- `popt` - fit parameters
- `tsamp` - sampling time (s)
- `title` - title of the plot
  

**Returns**:


<a name="burstfit.utils.plotter.plot_fit_results"></a>
#### plot\_fit\_results

```python
def plot_fit_results(sgram, function, popt, tsamp, fstart, foff, mask=None, outsize=None, title=None, show=True, save=False, outname="2d_fit_res", outdir=None, vmin=None, vmax=None)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/utils/plotter.py#L143)

**Arguments**:

- `sgram` - Original spectrogram data
- `function` - spectrogram function used for modeling
- `popt` - parameters for function
- `tsamp` - sampling time (s)
- `fstart` - start frequency (MHz)
- `foff` - channel bandwidth (MHz)
- `mask` - channel mask array
- `outsize` - resize the 2D plots
- `title` - title of the plot
- `show` - to show the plot
- `save` - to save the plot
- `outname` - output name of png file
- `outdir` - output directory for the plot
- `vmin` - minimum range of colormap
- `vmax` - maximum range of colormap
  

**Returns**:


<a name="burstfit.utils.plotter.plot_me"></a>
#### plot\_me

```python
def plot_me(datar, xlabel=None, ylabel=None, title=None)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/utils/plotter.py#L269)

Generic function to plot 1D or 2D array.
Requires SciencePlots.

**Arguments**:

- `datar` - data to plot
- `xlabel` - label of x axis
- `ylabel` - label of y axis
- `title` - title of the plot
  

**Returns**:


<a name="burstfit.utils.plotter.plot_mcmc_results"></a>
#### plot\_mcmc\_results

```python
def plot_mcmc_results(samples, name, param_starts, labels, save=False)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/utils/plotter.py#L301)

Save corner plot of MCMC results

**Arguments**:

- `samples` - MCMC samples to plot
- `name` - output name
- `param_starts` - mark the initial parameter guess
- `labels` - labels for axes
- `save` - to save the corner plot
  

**Returns**:


<a name="burstfit.utils.plotter.autocorr_plot"></a>
#### autocorr\_plot

```python
def autocorr_plot(n, y, name, save)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/utils/plotter.py#L342)

Make the autocorrelation plot to visualize convergence of MCMC.

**Arguments**:

- `n` - iterations
- `y` - autocorrelations
- `name` - outname of plot
- `save` - to save the plot
  

**Returns**:


