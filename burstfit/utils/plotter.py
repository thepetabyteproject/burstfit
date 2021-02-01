import matplotlib
import pylab as plt

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
        sgram, function, popt, title=None, show=True, save=False, outname="2d_fit_res"
):
    """
    Plot the result of spectrogram fit

    Args:
        sgram: input 2D array of spectrogram
        function: spectrogram function used for fitting
        popt: fit parameters
        title: title of the plot

    Returns:

    """
    model = function([0], *popt)
    if len(model.shape) == 1:
        model = model.reshape(sgram.shape)
    vmin = sgram.min()
    vmax = sgram.max()
    diff = sgram - model
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    axes[0, 0].imshow(sgram, aspect="auto", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Original Spectrogram")
    axes[0, 1].imshow(model, aspect="auto")
    axes[0, 1].set_title("Model")
    axes[1, 0].imshow(diff, aspect="auto", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Residual Spectrogram")
    axes[1, 1].plot(sgram.mean(0), c="k", alpha=0.7, linestyle="--", label="Original")
    axes[1, 1].plot(diff.mean(0), c="r", alpha=0.7, label="Residual")
    axes[1, 1].legend()
    axes[1, 1].set_title("Profiles")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig(outname + ".png", bbox_inches="tight")
    if show:
        plt.show()
    return fig


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
