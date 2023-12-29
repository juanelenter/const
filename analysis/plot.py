import matplotlib.pyplot as plt
import torch


def calibration_plot(metrics: dict[torch.Tensor], bins, title="", filename=None, labels=None):
    fig, ax = plt.subplots(figsize=(5, 5))

    width = (bins[1] - bins[0]) / (len(metrics) + 1)
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        correction_for_centered = width * (len(metrics) - 1) / 2
        bin_offset = i * width
        ax.bar(
            x=bin_midpoints + bin_offset - correction_for_centered,
            height=metric_values,
            label=labels[i],
            color=f"C{i}",
            alpha=0.5,
            width=width,
        )

    for j in range(len(bins) - 1):
        # TODO: instead of a line on the midpoint, could do a shaded square over all
        # of the range of the bin.
        y_value = bin_midpoints[j]
        ax.hlines(y=y_value, xmin=bins[j], xmax=bins[j + 1], color="red")

    # TODO: append a histogram with the number of samples in each bin.
    # and ensure that the x axis are aligned for both plots.

    plt.suptitle(title)
    ax.legend()
    ax.set_xlabel("Confidence on Correct Class")
    ax.set_xticks(bins)
    ax.set_ylabel("Accuracy")

    if filename:
        plt.savefig(f"{filename}.png")


def bar_plot(metrics: dict[torch.Tensor], title="", filename=None, labels=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        ax.bar(
            x=range(len(metric_values)),
            height=metric_values,
            label=labels[i],
            color=f"C{i}",
            alpha=0.5,
        )

    plt.suptitle(title)
    ax.legend()

    if filename:
        plt.savefig(f"{filename}.png")


def loss_hist(metrics: dict[list[torch.Tensor]], title="", num_bins=25, filename=None, labels=None):
    """Expects a dictionary. Each entry corresponds to a run, and contains a list of
    tensors with the loss for each group"""
    num_hists = len(list(metrics.items())[0][1])
    fig, axs = plt.subplots(ncols=num_hists, figsize=(5 * num_hists, 4))

    # GRID
    # Indicate proportion of the class ()
    # Add vertical line for epsilon

    min_value, max_value = 0.0, 0.0
    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        assert len(metric_values) == num_hists
        for j in range(num_hists):
            this_values = metric_values[j]
            min_value = min(min_value, this_values.min())
            max_value = max(max_value, this_values.max())
            axs[j].hist(this_values, bins=num_bins, label=labels[i], alpha=0.7, density=True, log=True)

    for i, ax in enumerate(axs):
        ax.set_ylabel("Density (Log scale)")
        ax.set_xlabel("Loss")
        ax.set_title(f"Group {i}")
        ax.set_xlim(min_value, max_value)

    plt.suptitle(title)

    if filename:
        plt.savefig(f"{filename}.png")


def plot_loss_vs_multiplier(loss_tensor, multiplier_tensor, split=""):
    pass
