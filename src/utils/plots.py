import datetime
import warnings

# Used to suppress warning for umap module import
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import matplotlib
import matplotlib.pyplot as plt
import torch
import umap.umap_ as umap
from matplotlib.colors import ListedColormap
from sklearn import svm

import shared

logger = shared.fetch_main_logger()


def generate_MNIST_plots(exp_cmp, model, loaded_datasets, config, val_accuracy, device):
    model.eval()

    if hasattr(exp_cmp, "evaluate_multipliers"):
        all_idx = torch.arange(loaded_datasets.train.dataset.data.shape[0], device=device)
        multiplier_values = exp_cmp.evaluate_multipliers(all_idx).detach().cpu().numpy()

        # Normalize the multiplier values for cleaner plotting
        min_size, max_size = multiplier_values.min() + 1e-8, multiplier_values.max()
        multiplier_values = 100 * (multiplier_values - min_size) / (max_size - min_size)
    else:
        multiplier_values = None

    X_numpy = loaded_datasets.train.dataset.data.cpu().numpy().reshape(-1, 28 * 28)
    y = loaded_datasets.train.dataset.targets.cpu().numpy()

    logger.info("Performing UMAP dimensionality reduction on MNIST...")
    umap_init_time = datetime.datetime.now()
    umap_2d = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2)
    X_2d = umap_2d.fit_transform(X_numpy)
    umap_time = (datetime.datetime.now() - umap_init_time).total_seconds()
    logger.info(f"UMAP completed in {umap_time} seconds.")

    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], marker="+", c="gray", s=5, alpha=0.1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10), s=multiplier_values, alpha=0.5)
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)

    if config.results.save_path is not None:
        file_name = config.results.save_path + f"mnist_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
        plt.savefig(file_name, dpi=600)

    if config.results.show_plots:
        plt.show()


def plot_hyperplane(w, b, X, plot_paralles=False):
    xx = np.linspace(X[:, 0].min(), X[:, 0].max())

    yy = (-w[0] / w[1]) * xx - (b / w[1])
    plt.plot(xx, yy, "k-")

    if plot_paralles:
        for sign in [-1, 1]:
            plane = yy + sign * (1 / w[1])
            plt.plot(xx, plane, "k--")


def plot_max_margin(X, y, plot_parallels=False):
    clf = svm.SVC(kernel="linear", C=1e8)
    clf.fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    plot_hyperplane(w, b, X, plot_paralles=True)


def generate_plots_for_2d_experiment(exp_cmp, model, loaded_datasets, config, val_accuracy, device):
    model.eval()

    if hasattr(exp_cmp, "evaluate_multipliers"):
        all_idx = torch.arange(loaded_datasets.train.dataset.data.shape[0], device=device)
        multiplier_values = exp_cmp.evaluate_multipliers(all_idx).detach().cpu().numpy()
    else:
        multiplier_values = None

    if config.cmp.pointwise_probability is not None:
        contour_levels = [config.cmp.pointwise_probability, 1 - config.cmp.pointwise_probability]
    else:
        contour_levels = []

    # Plot the decision boundary and show the training data
    plot_2d_decision_boundary(
        model,
        loaded_datasets.train.dataset.data,
        loaded_datasets.train.dataset.targets,
        multiplier_values,
        contour_levels=contour_levels,
        up_title=exp_cmp.__class__.__name__,  # Use the type of CMP as the plot title
        low_title=f"Val Acc: {(val_accuracy)*100:.2f}%",
        save_path=config.results.save_path,
        show=config.results.show_plots,
        filename_prefix="train_",
        show_max_margin=config.results.show_max_margin,
    )

    # Same plot with the val data
    plot_2d_decision_boundary(
        model,
        loaded_datasets.val.dataset.data,
        loaded_datasets.val.dataset.targets,
        None,
        contour_levels=contour_levels,
        up_title="Val Set - " + exp_cmp.__class__.__name__,  # Use the type of CMP as the plot title
        low_title=f"Val Acc: {(val_accuracy)*100:.2f}%",
        save_path=config.results.save_path,
        show=config.results.show_plots,
        filename_prefix="val_",
    )


def plot_2d_decision_boundary(
    model,
    X,
    y,
    example_size=None,
    contour_levels=[],
    up_title="",
    low_title="",
    save_path=None,
    show=False,
    filename_prefix="",
    show_max_margin=False,
):
    """
    Plots the decision boundary of a 2D model for a binary classification problem.
    """

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    samples_per_dim = 150
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    x_linspace = torch.linspace(x_min, x_max, samples_per_dim)
    y_linspace = torch.linspace(y_min, y_max, samples_per_dim)
    xx, yy = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
    Z = model(torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1).to(DEVICE))

    # Keep probability for class 1
    Z = torch.softmax(Z, dim=1)[:, 1].detach().cpu().numpy()

    Z = Z.reshape(xx.shape)
    cmap = ListedColormap(["#FF0000", "#0000FF"])

    fig, ax = plt.subplots()
    cf = ax.contourf(xx, yy, Z, cmap="RdBu", alpha=0.2)

    if show_max_margin:
        plot_max_margin(X, y, plot_parallels=True)

    # Always show contour levels at and around the decision boundary
    base_contour_levels = [0.4, 0.5, 0.6]
    contour = ax.contour(
        xx, yy, Z, levels=base_contour_levels, colors="black", alpha=0.4, linestyles="dashed", linewidths=0.5
    )
    plt.clabel(contour, inline=True, fontsize=4)

    # Show any extra contour levels requested
    custom_contour_levels = sorted(list(set(contour_levels) - set(base_contour_levels)))
    if len(custom_contour_levels) > 0:
        contour = ax.contour(
            xx, yy, Z, levels=custom_contour_levels, colors="black", alpha=0.4, linestyles="dashed", linewidths=1
        )
        plt.clabel(contour, inline=True, fontsize=7)

    # Show provided datapoints color-coded by class
    plt.scatter(X[:, 0], X[:, 1], marker="+", c=y, cmap=cmap, s=1, alpha=0.2)

    if example_size is not None:
        min_size, max_size = example_size.min() + 1e-8, example_size.max()
        if max_size == 0:
            low_title += " (**no SVs**)"
        else:
            example_size = 100 * (example_size - min_size) / (max_size - min_size)
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=example_size, alpha=0.5, edgecolors=(0, 0, 0, 0.5))

    plt.suptitle(up_title, fontsize=11)
    plt.title(low_title, fontsize=9)

    if save_path is not None:
        file_name = filename_prefix + f"decision_boundary_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
        plt.savefig(save_path + file_name, dpi=600)

    if show:
        plt.show()
