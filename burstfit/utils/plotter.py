import matplotlib
import pylab as plt

matplotlib.use("Agg")


def plot_1d_fit(
    xdata, ydata, function, popt, xlabel=None, ylabel=None, title=None, param_names=[]
):
    if len(param_names):
        if len(param_names) == len(popt):
            label = ""
            for i, param in enumerate(param_names):
                label += param + " = " + str(popt[i]) + "\n"
        else:
            label = f"fit: {popt}"
    else:
        label = f"fit: {popt}"

    plt.figure(figsize=(10, 8))
    plt.plot(xdata, ydata, "b-", label="data")
    plt.plot(
        xdata,
        function(xdata, *popt),
        "g--",
        label=label,
    )
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    return plt.show()


def plot_2d_fit(sgram, function, popt, param_names=[]):
    model = function([0], *popt)
    if len(model.shape) == 1:
        model = model.reshape(sgram.shape())
    diff = sgram - model
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    axes[0].imshow(sgram, aspect="auto")
    axes[0].set_title("Original Spectrogram")
    axes[1].imshow(diff, aspect="auto")
    axes[1].set_title("Residual Spectrogram")
    axes[2].plot(sgram.mean(0), c="k", alpha=0.7, linestyle="--", label="Original")
    axes[2].plot(diff.mean(0), c="r", alpha=0.7, label="Residual")
    axes[2].legend()
    axes[2].set_title("Profiles")
    return fig


def plot_me(datar, xlabel=None, ylabel=None, title=None):
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
