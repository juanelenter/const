import logging
import os
import sys
from functools import partial

import dotenv
import plot
import torch
from wandb_utils import fetch_models_and_configs

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import shared
from src import datasets, utils

dotenv.load_dotenv()

logger = logging.getLogger()
# Configure root logger to use rich logging
shared.configure_logger(logger, level=logging.INFO)

logging.basicConfig()
logger = logging.getLogger()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_dataloaders(config):
    dataset_namespace, dataset_metadata = datasets.build_datasets(config, True)

    # TODO: ensure sequential sampling
    dataloader_namespace = datasets.build_dataloaders(dataset_namespace, config, DEVICE, {})

    return dataloader_namespace


@torch.inference_mode()
def compute_predictions(model, dataloader):
    model.eval()
    model.to(DEVICE)

    all_predictions = []
    all_targets = []

    with torch.inference_mode():
        for batch in dataloader:
            inputs, targets, indices = batch

            inputs = inputs.to(DEVICE)
            predictions = model(inputs)

            all_predictions.append(predictions)
            all_targets.append(targets)

    all_predictions = torch.cat(all_predictions).cpu()
    all_targets = torch.cat(all_targets)

    return all_predictions, all_targets


# TODOs(juan43ramirez): loss and multiplier histograms per group


def calibration_error(predictions, targets, num_bins=15):

    per_sample_probability = predictions.softmax(dim=1)[torch.arange(len(predictions)), targets]
    per_sample_accuracy = (predictions.argmax(dim=1) == targets).float()

    bins = torch.linspace(0.0, 1.0, num_bins + 1)

    # TODO: right=True means
    bin_indices = torch.bucketize(per_sample_probability, bins, right=True)
    accuracies_per_bin = []

    for i in range(num_bins):
        accuracies_per_bin.append(per_sample_accuracy[bin_indices == i].mean())

    return torch.stack(accuracies_per_bin), bins


def grouped_metric(predictions, targets, metric, groups=None, aggregate=False):
    values = metric(predictions, targets)

    if groups is None:
        groups = targets

    values_per_group = []
    for group in groups.unique():
        values_per_group.append(values[groups == group])

    if aggregate:
        values_per_group = torch.stack([g.mean() for g in values_per_group])

    return values_per_group


if __name__ == "__main__":

    # ----------------------------------------------------------------------------------
    # ------------------------- Choose runs to compare ---------------------------------
    # ----------------------------------------------------------------------------------
    # # CIFAR10 no long tail
    # entity = "jgp"
    # project = "feasible.cifar10.LT"
    # run_ids = ["inifhxju", "pxm8u0mi"]
    # labels = ["ERM", "FL"]

    # # MNIST
    # entity = "jgp"
    # project = "feasible-learning"
    # run_ids = ["3550381", "3550384"]
    # labels = ["ERM", "FL-90"]

    # ----------------------------------------------------------------------------------
    # ------------------------- Choose metric to compare -------------------------------
    # ----------------------------------------------------------------------------------
    # # Loss histogram per group
    # name = "loss_hist"
    # metric = partial(torch.nn.functional.cross_entropy, reduction="none")
    # func = partial(grouped_metric, metric=metric)
    # plot_fn = plot.loss_hist
    # kwargs = {}

    # # Accuracy per group
    # name = "accuracy_per_group"
    # metric = lambda predictions, targets: (predictions.argmax(dim=1) == targets).float()
    # func = partial(grouped_metric, metric=metric, aggregate=True)
    # plot_fn = plot.bar_plot
    # kwargs = {}

    # # Calibration
    # name = "Calibration"
    # func = calibration_error
    # plot_fn = plot.calibration_plot
    # kwargs = {"num_bins": 10}
    # plot_kwargs = {}

    # TODO(juan43ramirez): for calibration plots:
    # Number/proportion of samples per bin

    # TODO(juan43ramirez): getting all the models at the same time requires a lot of
    # memory. Could fetch them one at a time.
    configs, models = fetch_models_and_configs(run_ids, entity, project)
    figs_dir = os.environ["FIGS_DIR"]
    os.makedirs(figs_dir, exist_ok=True)

    assert all(configs[run_ids[0]].data == config.data for config in configs.values())
    dataloaders = build_dataloaders(configs[run_ids[0]])

    measurement = {}
    for split, dataloader in utils.scan_namespace(dataloaders):
        measurement[split] = {}

        for run_id, model in models.items():
            predictions, targets = compute_predictions(model, dataloader)
            metric_values = func(predictions, targets, **kwargs)

            if name == "Calibration":
                metric_values, bins = metric_values
                plot_kwargs["bins"] = bins

            measurement[split][run_id] = metric_values

        # plot.bar_plot(group_accuracies[split], title=f"Accuracy per group ({split})")
        filename = f"{figs_dir}/{name}_{split}"
        title = f"{name} ({split})"
        plot_fn(metrics=measurement[split], title=title, filename=filename, labels=labels, **plot_kwargs)
