import os

import corner
import matplotlib
import numpy as np
import pylab as plt
from skimage.transform import resize

matplotlib.use("Agg")


def plot_1d_fit(
    xdata,
    ydata,
    function,
    popt,
    xlabel=None,
    ylabel=None,
    title=None,
    param_names=[],
    show=True,
    save=False,
    outname="1d_fit_res",
):
    """
    Plot the results of 1D fits

    Args:
        xdata: x value array
        ydata: original data values
        function: function used for fitting
        popt: fit parameters of the function
        xlabel: label of x axis
        ylabel: label of y axis
        title: title of the plot
        param_names: names of the parameters

    Returns:

    """
    if len(param_names):
        if len(param_names) == len(popt):
            label = ""
            for i, param in enumerate(param_names):
                label += param + " = " + str(popt[i]) + "\n"
        else:
            label = f"fit: {popt}"
    else:
        label = f"fit: {popt}"

    plt.figure(figsize=(8, 6))
    plt.plot(xdata, ydata, "b-", label="data")
    plt.plot(
        xdata,
        function(xdata, *popt),
        "g--",
        label=label,
    )
    plt.tight_layout()
    plt.legend()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if save:
        plt.savefig(outname + ".png", bbox_inches="tight")
    if show:
        plt.show()


def plot_2d_fit(
    sgram,
    function,
    popt,
    tsamp,
    title=None,
    show=True,
    save=False,
    outname="2d_fit_res",
    outdir=None,
):
    """
    Plot the result of spectrogram fit

    Args:
        sgram: input 2D array of spectrogram
        function: spectrogram function used for fitting
        popt: fit parameters
        tsamp: sampling time (s)
        title: title of the plot

    Returns:

    """
    model = function([0], *popt)
    if len(model.shape) == 1:
        model = model.reshape(sgram.shape)

    vmin = sgram.min()
    vmax = sgram.max()

    diff = sgram - model
    nf, nt = sgram.shape

    l = np.linspace(-nt // 2, nt // 2, nt)
    ts = l * tsamp * 1000

    fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
    axes[0, 0].imshow(
        sgram, aspect="auto", vmin=vmin, vmax=vmax, extent=[ts[0], ts[-1], nf, 0]
    )
    axes[0, 0].set_title("Original Spectrogram")
    axes[0, 1].imshow(model, aspect="auto", extent=[ts[0], ts[-1], nf, 0])
    axes[0, 1].set_title("Model")
    axes[1, 0].imshow(
        diff, aspect="auto", vmin=vmin, vmax=vmax, extent=[ts[0], ts[-1], nf, 0]
    )
    axes[1, 0].set_title("Residual Spectrogram")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 1].plot(
        ts, sgram.mean(0), c="k", alpha=0.7, linestyle="--", label="Original"
    )
    axes[1, 1].plot(
        ts, diff.mean(0), c="r", alpha=0.7, linestyle="dotted", label="Residual"
    )
    axes[1, 1].legend()
    axes[1, 1].set_title("Profiles")
    axes[1, 1].set_xlabel("Time (ms)")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save:
        if not outdir:
            outdir = os.getcwd()
        plt.savefig(outdir + "/" + outname + ".png", bbox_inches="tight", dpi=300)
    if show:
        plt.show()


def plot_fit_results(
    sgram,
    function,
    popt,
    tsamp,
    fstart,
    foff,
    mask=None,
    outsize=None,
    title=None,
    show=True,
    save=False,
    outname="2d_fit_res",
    outdir=None,
    vmin=None,
    vmax=None,
):
    """

    Args:
        sgram: Original spectrogram data
        function: spectrogram function used for modeling
        popt: parameters for function
        tsamp: sampling time (s)
        fstart: start frequency (MHz)
        foff: channel bandwidth (MHz)
        mask: channel mask array
        outsize: resize the 2D plots
        title: title of the plot
        show: to show the plot
        save: to save the plot
        outname: output name of png file
        outdir: output directory for the plot
        vmin: minimum range of colormap
        vmax: maximum range of colormap

    Returns:

    """
    if np.any(outsize):
        sgram[sgram.mask] = 0

    model = function([0], *popt)
    if len(model.shape) == 1:
        model = model.reshape(sgram.shape)

    nf, nt = sgram.shape

    freqs = fstart + foff * np.linspace(0, nf - 1, nf)
    if not np.any(mask):
        mask = np.zeros(len(freqs), dtype=np.bool)

    if np.any(outsize):
        assert len(outsize) == 2
        assert nf % outsize[0] == 0
        tsamp = tsamp * nt / outsize[1]
        freqs = freqs.reshape(outsize[0], nf // outsize[0]).mean(-1)
        mask = (mask.reshape(outsize[0], nf // outsize[0]).mean(-1)).astype(np.bool)
        model = resize(model, outsize)
        sgram = resize(sgram, outsize)

    if not vmin:
        vmin = sgram.min()

    if not vmax:
        vmax = sgram.max()

    diff = sgram - model
    nf, nt = sgram.shape
    l = np.linspace(-nt // 2, nt // 2, nt)
    ts = l * tsamp * 1000

    fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey="row")
    kwargs = {}
    kwargs["extent"] = [ts[0], ts[-1], freqs[-1], freqs[0]]
    kwargs["vmin"] = vmin
    kwargs["vmax"] = vmax
    kwargs["aspect"] = "auto"

    axes[0, 0].imshow(sgram, **kwargs)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(model, aspect="auto", extent=[ts[0], ts[-1], freqs[-1], freqs[0]])
    axes[0, 1].set_title("Model")
    axes[0, 2].imshow(diff, **kwargs)
    axes[0, 2].set_title("Residual")
    for freq in zip(freqs[mask]):
        for ax in axes[0, :]:
            ax.axhline(freq, color="r", xmin=0, xmax=0.03, lw=0.1)

    axes[1, 0].plot(
        ts, sgram.mean(0), c="r", alpha=1, linestyle="solid", label="Original"
    )
    axes[1, 1].plot(
        ts,
        model.mean(0),
        c="r",
        alpha=1,
        linestyle="solid",
        label="Residual",
    )
    axes[1, 2].plot(
        ts,
        diff.mean(0),
        c="r",
        alpha=1,
        linestyle="solid",
        label="Residual",
    )
    axes[1, 1].set_ylim()
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[1, 2].set_xlabel("Time (ms)")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save:
        if not outdir:
            outdir = os.getcwd()
        plt.savefig(
            outdir + "/" + outname + "fit_results.png", bbox_inches="tight", dpi=300
        )
    if show:
        plt.show()


def plot_me(datar, xlabel=None, ylabel=None, title=None):
    """
    Generic function to plot 1D or 2D array.
    Requires SciencePlots.

    Args:
        datar: data to plot
        xlabel: label of x axis
        ylabel: label of y axis
        title: title of the plot

    Returns:

    """
    with plt.style.context(["notebook"]):
        if len(datar.shape) == 1:
            plt.plot(datar)
        else:
            plt.imshow(
                datar,
                aspect="auto",
            )
            plt.colorbar()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)
    return plt.show()


def plot_mcmc_results(samples, name, param_starts, labels, save=False):
    """
    Save corner plot of MCMC results

    Args:
        samples: MCMC samples to plot
        name: output name
        param_starts: mark the initial parameter guess
        labels: labels for axes
        save: to save the corner plot

    Returns:

    """
    ndim = samples.shape[1]
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.suptitle(name)

    plt.figure()
    fig = corner.corner(
        samples,
        labels=labels,
        bins=20,
        truths=param_starts,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    fig.suptitle(name)
    plt.tight_layout()
    if save:
        fig.savefig(f"{name}_corner.png", bbox_inches="tight")


def autocorr_plot(n, y, name, save):
    """
    Make the autocorrelation plot to visualize convergence of MCMC.

    Args:
        n: iterations
        y: autocorrelations
        name: outname of plot
        save: to save the plot

    Returns:

    """
    plt.figure()
    plt.plot(n, n / 100.0, "--k")
    plt.plot(n, y)
    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$")
    plt.title(f'{name}')
    if save:
        plt.savefig(f"{name}_autocorr.png", bbox_inches="tight")
