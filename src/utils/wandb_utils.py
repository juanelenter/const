import os

import dotenv
import numpy as np
import torch
import wandb

dotenv.load_dotenv()


def make_wandb_histogram(values: torch.Tensor, bins=50):
    # TODO(juan43ramirez): is it necessary to move the tensor to the cpu here?
    values = values.flatten().detach().cpu().numpy()
    hist = np.histogram(values, bins=bins)
    return wandb.Histogram(np_histogram=hist)


if __name__ == "__main__":

    """Test logging a wandb histogram."""

    run = wandb.init(project="test", dir=os.environ["WANDB_DIR"])

    for i in range(10):
        values = np.random.randn(1000) + i
        histogram = make_wandb_histogram(values)
        wandb.log({"histogram": histogram}, step=i)
