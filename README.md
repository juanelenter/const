# feasible-learning


## Usage

To run a single ImageNet experiment:
- Activate your environment
- `source scripts/copy_datasets.sh`
- `copy_dataset imagenet . 0` (Copy the (full) ImageNet dataset to `SLURM_TMPDIR`)
- `bash scripts/imagenet.sh`

**An example of a full command:** Read on for more details.
```
python main.py \
    --model_config=configs/model.py:"model=MNIST_CNN" \
    --data_config=configs/data.py:"data=mnist" \
    --task_config=configs/task.py:"task=feasibility task.pointwise_probability=0.9 task.multiplier_kwargs.restart_on_feasible=False optim.dual_optimizer.shared_kwargs.lr=1e-5" \
    --config.train.total_epochs=20 \
    --metrics_config=configs/metrics.py:classification \
    --resources_config=configs/resources.py:"cluster=debug tasks_per_node=1 use_ddp=False" \
    --config.logging.wandb_mode=disabled
```

**2D linearly separable dataset**
```
python main.py \
    --model_config=configs/model.py:"model=TwoDim_MLP model.init_kwargs.hidden_sizes=(5,) optim.primal_optimizer.name=Adam optim.primal_optimizer.shared_kwargs.lr=7e-1 optim.primal_optimizer.shared_kwargs.weight_decay=0.0" \
    --data_config=configs/data.py:"data=linsep_2d data.dataset_kwargs.train_samples=128 data.dataset_kwargs.val_samples=1000" \
    --task_config=configs/task.py:"task=feasibility task.pointwise_probability=0.99 task.multiplier_kwargs.restart_on_feasible=False optim.dual_optimizer.shared_kwargs.lr=1e-3 task.early_stop_on_feasible=False" \
   --config.train.total_epochs=100 \
    --metrics_config=configs/metrics.py:classification \
    --resources_config=configs/resources.py:"cluster=debug tasks_per_node=1 use_ddp=False" \
    --config.logging.wandb_mode=online
```

**Two moons dataset**
```
python main.py \
    --model_config=configs/model.py:"model=TwoDim_MLP model.init_kwargs.hidden_sizes=(50, 50, 50)" \
    --data_config=configs/data.py:"data=two_moons data.dataset_kwargs.train_samples=100 data.dataset_kwargs.val_samples=2000" \
    --task_config=configs/task.py:"task=feasibility task.pointwise_probability=0.9 task.multiplier_kwargs.restart_on_feasible=False" \
    --config.train.total_epochs=1000 \
    --metrics_config=configs/metrics.py:classification \
    --resources_config=configs/resources.py:"cluster=debug tasks_per_node=1 use_ddp=False" \
    --config.logging.wandb_mode=online
```


Note that the parsed text arguments for file-based configs need to point to the "full
path" of the configuration item. For example, if you want to change `restart_on_feasible`
in `config.task.multiplier_kwargs`, you need to use
`task.multiplier_kwargs.restart_on_feasible=BOOL` (and NOT
`multiplier_kwargs.restart_on_feasible=BOOL`).

| Cluster | Is local? | Is SLURM? | Is interactive? | Is background? | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `debug` | ✅ | 🚫 | ✅ | 🚫 | For local execution on your machine or on a compute node |
| `local` | ✅ | 🚫 | 🚫 | ✅ | This could be locally on your machine or on a compute node |
| `unkillable` | 🚫 | ✅ | 🚫 | ✅ | To run jobs on SLURM `unkillable` partition |
| `main` | 🚫 | ✅ | 🚫 | ✅ | To run jobs on SLURM `main` partition |
| `long` | 🚫 | ✅ | 🚫 | ✅ | To run jobs on SLURM `long` partition |

- Keep in mind that you can run the code from a compute node on SLURM via `debug` or `local`.
- Currently, using multiple GPUs on `debug` is not supported.
- But you can use multiple GPUs on `local` or on a compute node. This job will run in the background (on your node).
- By default we do NOT use DDP. If you want to use DDP, you must pass `use_ddp=True` to `resources_config`. See below.

The string to be passed to `resources_config=configs/resources.py:STRING` is of the form
`"cluster=CLUSTER tasks_per_node=INT use_ddp=BOOL"`. Note that spaces are used as separators.
For example, if you want to run on `local` (your machine or compute node) with 2 GPUs
and DDP, you would use:
```
    --resources_config=configs/resources.py:"cluster=debug tasks_per_node=2 use_ddp=True"
```

Examples of valid resource configurations:
- `"cluster=debug"` # This can be used to debug your code locally or on a compute node
- `"cluster=local tasks_per_node=1 use_ddp=False"` # This would only use 1 GPU even if more are available
- `"cluster=local tasks_per_node=2 use_ddp=True"`
- `"cluster=unkillable"`
- `"cluster=main tasks_per_node=2 use_ddp=True"`
- `"cluster=long tasks_per_node=2 use_ddp=True"`


## Required enviroment variables

We use [`dotenv`](https://github.com/theskumar/python-dotenv) to manage environment variables. Please create a `.env` file in the root directory of the project and add the following variables:

```
# Location of the directory containing your datasets
DATA_DIR=

# The directory where the results will be saved
CHECKPOINT_DIR=

# If you want to use Weights & Biases, add the entity name here
WANDB_ENTITY=

# Directory for Weights & Biases local storage
WANDB_DIR=

# Directory for logs created by submitit
SUBMITIT_DIR=
```

## Code style

- This project uses `black`, `isort`, and `flake8` for enforcing code style. See `requirements.txt` for version numbers.
- We use `pre-commit` hooks to ensure that all code committed respects the code style.
- After (1) cloning the repo, (2) creating your environment and (3) installing the required
packages, you are strongly encouraged to run `pre-commit install` to set-up pre-commit hooks.

### Logging format

Whenever you are using logging inside a module, please remember to use the _rich_ formatting.

Do NOT do this:
```
import logging
logger = logging.getLogger(__name__)
```

DO this instead:
```
import shared
logger = shared.fetch_main_logger()
```
