import logging
import os
import pickle
import time
from operator import attrgetter
from pathlib import Path
from types import SimpleNamespace

import cooper
import submitit
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel

import shared
from src import constrained, datasets, models, utils

logger = shared.fetch_main_logger(apply_basic_config=True)


class Trainer:
    def __init__(self, config):
        self.config = config

        logger.info("Initialized trainer with configuration:")
        logger.info(config)
        logger.info(f"Current working directory is {os.getcwd()}")

    def __call__(self):
        self._make_reproducible()

        self.dist = utils.distributed.init_distributed(self.config.resources)
        self.device = self.dist.device
        self.is_main_process = self.dist.rank == 0

        # Update the trainer logger to include rank information for multi-GPU training
        self._update_logger()

        self.wandb_run, self.run_checkpoint_dir, self.is_wandb_run_resumed = self._create_wandb_logger()

        logger.info("Trainer called with config:")
        logger.info(self.config)

        self.model, self.model_without_ddp = self._create_model()

        self.dataset_metadata, self.dataloaders = self._create_dataloaders()

        self.num_samples = utils.extract_to_namespace(self.dataloaders, extract_fn=lambda loader: len(loader.dataset))
        self.num_batches = utils.extract_to_namespace(self.dataloaders, extract_fn=lambda loader: len(loader))

        self.num_steps = self._init_stopping_condition()
        self.eval_period_steps = self._init_evaluation_period()

        self.cmp = self._create_cmp()
        self.cooper_optimizer, self.schedulers = self._create_optimizers_and_schedulers()

        self.metrics = self._create_metrics()

        if self.config.data.dataset_kwargs.get("use_data_augmentation", False):
            if self.config.task.multiplier_kwargs.restart_on_feasible:
                raise ValueError("Dual restarts should not be combined with data augmentation!")

        self.train()

        self.generate_reports()

        self._clean_finish()

    def _update_logger(self):
        shared.configure_logger(
            logger=shared.fetch_main_logger(),
            custom_format=f"(Rank:{self.dist.rank}/WS:{self.dist.world_size}) %(module)s:%(funcName)s:%(lineno)d | %(message)s ",
            level=getattr(logging, self.config.logging.log_level),
            show_path=self.config.logging.wandb_mode == "disabled",  # Only show path hyperlinks if not using wandb
        )

    def _make_reproducible(self):
        utils.set_seed(self.config.train.seed)
        if self.config.train.use_deterministic_ops:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _create_dataloaders(self):
        dataset_namespace, dataset_metadata = datasets.build_datasets(self.config, self.is_main_process)
        dataloader_namespace = datasets.build_dataloaders(dataset_namespace, self.config, self.device, self.dist)
        return dataset_metadata, dataloader_namespace

    def _create_model(self):
        torch.manual_seed(self.config.model.init_seed)

        try:
            model_class = getattr(models, self.config.model.name)
        except AttributeError:
            raise ValueError(f"Unknown model: {self.config.model.name}")

        model = model_class(**self.config.model.init_kwargs)
        model.to(device=self.device)
        model_without_ddp = model
        param_count = sum([torch.prod(torch.tensor(p.shape)).item() for p in model.parameters()])
        logger.info(f"Created model {self.config.model.name} with " + f"{param_count} parameters")

        if self.config.resources.use_ddp:
            total_gpus = self.config.resources.tasks_per_node * self.config.resources.nodes

            if total_gpus < 1:
                raise ValueError(f"Requested using DDP but no GPUs available.")
            elif total_gpus == 1:
                logger.warning("Requested using DDP but only 1 GPU available. Continuing without DDP.")
            else:
                model = DistributedDataParallel(model)
                logger.info("Successfully wrapped model in DDP.")

        return model, model_without_ddp

    def _create_cmp(self):
        cmp_metadata = SimpleNamespace(dataset=self.dataset_metadata, is_main_process=self.is_main_process)
        cmp = constrained.build_cmp(
            config=self.config,
            device=self.device,
            num_training_samples=self.num_samples.train,
            cmp_metadata=cmp_metadata,
        )
        if self.is_main_process:
            wandb.config.update({"task.pointwise_loss_level": cmp.pointwise_loss_level()})

        return cmp

    def _create_optimizers_and_schedulers(self):
        return constrained.build_cooper_optimizer_and_schedulers(
            model=self.model_without_ddp, cmp=self.cmp, config=self.config
        )

    def _save_checkpoint(self):
        if self.is_main_process and self.config.checkpointing.enabled:
            os.makedirs(self.run_checkpoint_dir, exist_ok=True)

            # The `cooper_optimizer` checkpoint already containts the multiplier states
            # This can be accessed via `checkpoint["cooper_optimizer"].multiplier_states`
            checkpoint = {
                "model": self.model_without_ddp.state_dict(),
                "cooper_optimizer": self.cooper_optimizer.state_dict(),
                "steps_taken": self.steps_taken,
                "epoch": self.epoch,
                "elapsed_time": (time.time() - self.start_time) + self.elapsed_time,
            }

            filename = os.path.join(self.run_checkpoint_dir, "checkpoint.pt")
            torch.save(checkpoint, filename)
            wandb.save(filename, base_path=self.run_checkpoint_dir)
            logger.info(f"Saved checkpoint to {self.run_checkpoint_dir} (step={self.steps_taken}; epoch={self.epoch})")

    def _load_checkpoint(self):
        logger.info("Attempting to resume from checkpoint...")
        filename = os.path.join(self.run_checkpoint_dir, "checkpoint.pt")
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)

            self.model_without_ddp.load_state_dict(checkpoint["model"])

            # This also loads the multiplier states (via the constraint groups)
            self.cooper_optimizer = cooper.optim.utils.load_cooper_optimizer_from_state_dict(
                cooper_optimizer_state=checkpoint["cooper_optimizer"],
                primal_optimizers=self.cooper_optimizer.primal_optimizers,
                dual_optimizers=getattr(self.cooper_optimizer, "dual_optimizers", None),
                constraint_groups=getattr(self.cmp, "constraint_groups", None),
            )

            self.steps_taken = checkpoint["steps_taken"]
            self.epoch = checkpoint["epoch"]
            self.elapsed_time = checkpoint["elapsed_time"]
            self.start_time = time.time()
        else:
            raise ValueError("WandB run requested resuming but no checkpoint found.")

    def _create_metrics(self):
        metrics = SimpleNamespace(batch=self.config.metrics.batch_level, epoch=SimpleNamespace())

        logger.info(f"Metrics logged at each batch: {metrics.batch}")

        logger.info(f"Epoch-level metrics:")
        for split in self.config.metrics.epoch_level:
            metrics_in_split = {}
            for metric_config in getattr(self.config.metrics.epoch_level, split):
                kwargs = metric_config.kwargs if len(metric_config.kwargs) > 0 else {}
                if "PerClass" in metric_config.name:
                    kwargs = kwargs | {"num_classes": self.dataset_metadata.num_classes}

                metric_class = getattr(utils.metrics, metric_config.name)
                metrics_in_split[metric_config.log_name] = metric_class(log_name=metric_config.log_name, **kwargs)
                logger.info(f"\t Split: {split} \t Name: {metric_config.name} \t Kwargs: {kwargs}")

            metrics.epoch.__setattr__(split, metrics_in_split)
            logger.info(f"Instantiated {len(metrics_in_split)} {split} metrics")

        return metrics

    def _create_wandb_logger(self):
        run, run_checkpoint_dir, is_wandb_run_resumed = None, None, None
        if self.is_main_process:
            is_local_job = not ("SLURM_JOB_ID" in os.environ.keys() and self.config.resources.cluster == "slurm")

            # This is compatible with preemption since the SLURM_JOB_ID value is
            # preserved after preemption.
            custom_run_id = None if is_local_job else os.environ["SLURM_JOB_ID"]

            run = wandb.init(
                entity=os.environ["WANDB_ENTITY"],
                project=self.config.logging.wandb_project,
                dir=os.environ["WANDB_DIR"],
                id=custom_run_id,
                mode=self.config.logging.wandb_mode,
                resume="allow",
                tags=self.config.logging.wandb_tags,
            )
            logger.info(f"Initialized WandB run with id {run.id}")

            wandb.config.update(self.config.to_dict(), allow_val_change=True)

            # Define metrics for custom x-axis
            wandb.define_metric("batch/*", step_metric="_step")
            wandb.define_metric("_epoch")
            wandb.define_metric("val/*", step_metric="_epoch")
            wandb.define_metric("train/*", step_metric="_epoch")
            wandb.define_metric("lambda/*", step_metric="_epoch")

            run_subdir = run.id if is_local_job else os.environ["SLURM_JOB_ID"]
            run_checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"]) / run_subdir
            logger.info(f"Checkpoints will be saved to {run_checkpoint_dir}")

            is_wandb_run_resumed = run.resumed

        is_wandb_run_resumed = utils.distributed.do_broadcast_object(is_wandb_run_resumed)
        run_checkpoint_dir = utils.distributed.do_broadcast_object(run_checkpoint_dir)

        return run, run_checkpoint_dir, is_wandb_run_resumed

    def _init_stopping_condition(self):
        train_config = self.config.train
        if train_config.total_epochs is not None and train_config.total_steps is not None:
            raise ValueError("Train config contains both 'total_epochs' and 'total_steps'. Please specift only one")
        elif train_config.total_steps is not None:
            num_steps = train_config.total_steps
        elif train_config.total_epochs is not None:
            num_steps = self.num_batches.train * train_config.total_epochs
        else:
            raise ValueError("No stopping condition was specified.")

        num_epochs = num_steps / self.num_batches.train
        logger.info(f"Training loop was configured to run for {num_steps} steps ({num_epochs: .2f} epochs)")

        return num_steps

    def _init_evaluation_period(self):
        eval_period_steps = self.config.logging.eval_period_steps
        eval_period_epochs = self.config.logging.eval_period_epochs

        if eval_period_steps is not None and eval_period_epochs is not None:
            raise ValueError("Train config should specify exactly one of 'eval_period_steps' and 'eval_period_epochs'.")
        if eval_period_steps:
            _eval_period_steps = eval_period_steps
        elif eval_period_epochs:
            _eval_period_steps = self.num_batches.train * eval_period_epochs
        else:
            raise ValueError("No evaluation period was specified.")

        _eval_period_epochs = _eval_period_steps / self.num_batches.train
        logger.info(f"Evaluation happening every {_eval_period_steps} steps ({_eval_period_epochs: .2f} epochs)")

        return _eval_period_steps

    def _format_logs_for_wandb(self, metrics: dict[str, float], prefix: str = "train/"):
        wandb_dict = {prefix + k: v for k, v in metrics.items()}
        wandb_dict["_epoch"] = self.epoch
        wandb_dict["wall_sec"] = self.elapsed_time + (time.time() - self.start_time)
        wandb_dict["training_steps"] = self.steps_taken

        return wandb_dict

    def _clean_finish(self):
        utils.distributed.wait_for_all_processes()

        if self.is_main_process:
            logger.info("Attempting to close WandB logger")
            wandb.finish()
            logger.info("Shutting down gracefully")

    def log_multiplier_stats(self):
        if self.is_main_process:
            multiplier_stats = self.cmp.extract_multiplier_stats()
            if multiplier_stats is not None:
                all_multiplier_values = multiplier_stats.pop("all_multiplier_values")
                all_multiplier_values = all_multiplier_values.half()

                multiplier_folder = f"{self.run_checkpoint_dir}/multipliers"
                file_path = os.path.join(multiplier_folder, f"epoch_{self.epoch}.pt")
                os.makedirs(multiplier_folder, exist_ok=True)
                torch.save(all_multiplier_values, file_path)
                wandb.save(file_path, base_path=self.run_checkpoint_dir)
                logger.info(f"Saved multiplier values to {file_path}")

                # Log multiplier stats and histogram to wandb
                multipier_histogram = utils.wandb_utils.make_wandb_histogram(all_multiplier_values)
                multiplier_stats = {f"lambda/{stat_name}": v for stat_name, v in multiplier_stats.items()}
                multiplier_stats["lambda/histogram"] = multipier_histogram

                wandb_multiplier_stats = self._format_logs_for_wandb(multiplier_stats, prefix="")
                wandb.log(wandb_multiplier_stats, step=self.steps_taken)

    def log_losses_on_split(self, split, all_losses):
        if self.is_main_process:
            # Sync the losses. Taking the max as each process will have 0s for
            # the samples it didn't see.
            all_losses = utils.distributed.do_all_gather_object(all_losses)
            if not self.dist.multi_gpu:
                all_losses = all_losses[0]
            else:
                # TODO(juan'): untested in an actual multi-gpu run
                all_losses = torch.max(torch.cat(all_losses, dim=0), dim=0)
            all_losses = all_losses.half()

            losses_folder = f"{self.run_checkpoint_dir}/losses/{split}"
            file_path = os.path.join(losses_folder, f"epoch_{self.epoch}.pt")
            os.makedirs(losses_folder, exist_ok=True)
            torch.save(all_losses, file_path)
            wandb.save(file_path, base_path=self.run_checkpoint_dir)
            logger.info(f"Saved {split} losses to {file_path}")

            loss_histogram = utils.wandb_utils.make_wandb_histogram(all_losses)
            log_dict = self._format_logs_for_wandb({"loss/histogram": loss_histogram}, prefix=f"{split}/")
            wandb.log(log_dict, step=self.steps_taken)

    @torch.inference_mode()
    def process_batch_for_evaluation(self, batch_data, split_metrics):
        inputs = batch_data[0].to(device=self.device, non_blocking=True)
        targets = batch_data[1].to(device=self.device, non_blocking=True)
        predictions = self.model(inputs)
        per_sample_loss = self.cmp.loss_fn(predictions, targets, per_sample=True)

        known_args = {
            "targets": targets,
            "predictions": predictions,
            "per_sample_loss": per_sample_loss,
            "cmp": self.cmp,
        }
        for metric in split_metrics.values():
            kwargs_for_metric = {key: known_args[key] for key in metric.forward_args}
            kwargs_for_metric["return_value"] = False
            # This call computes the metric values and updates the meters internally
            metric(**kwargs_for_metric)

        return per_sample_loss

    @torch.inference_mode()
    def _eval_loop(self):
        logger.info(f"Initiating evaluation loop on rank {self.dist.rank}")
        self.model.eval()

        self.log_multiplier_stats()

        is_all_train_feasible = False

        for split, split_metrics in utils.scan_namespace(self.metrics.epoch):
            # TODO(juan43ramirez): this method is too long. Consider refactoring.

            logger.info(f"Computing metrics for {split} split")

            split_meters = {}
            for metric in split_metrics.values():
                metric.reset_meters()
                split_meters[metric] = metric.meters

            if len(split_meters) == 0:
                continue

            # Evaluate batch-dependent metrics and update meters
            is_dataset_indexed = isinstance(getattr(self.dataloaders, split).dataset, datasets.IndexedDataset)
            all_losses = torch.zeros(getattr(self.num_samples, split), device=self.device)

            for batch_data in getattr(self.dataloaders, split):
                indices = batch_data[2].to(device=self.device, non_blocking=True) if is_dataset_indexed else None
                per_sample_loss = self.process_batch_for_evaluation(batch_data, split_metrics)
                if is_dataset_indexed:
                    all_losses[indices] = per_sample_loss

            utils.distributed.wait_for_all_processes()

            if self.is_main_process:
                # In the case of DDP, the main process will trigger the meters to be synced
                # across all processes. Once synced, we can aggregate the results and log
                # them to WandB.
                logger.info(f"Aggregating {split} metrics on rank {self.dist.rank}")

                val_log_dict = {}
                for metric in split_meters.keys():
                    logger.info(f"Aggregating values for metric {metric}")
                    for metric_known_return, meter in metric.meters.items():
                        # This triggers the sync across processes at the meter level
                        meter_values = meter.get_result_dict()

                        if len(meter_values) == 1 and "avg" in meter_values:
                            create_key_fn = lambda metric_kr, meter_kr: f"{metric_kr}"
                        else:
                            create_key_fn = lambda metric_kr, meter_kr: f"{metric_kr}/{meter_kr}"

                        for meter_known_return in meter_values.keys():
                            key = create_key_fn(metric_known_return, meter_known_return)
                            if key in val_log_dict:
                                raise ValueError(f"Duplicate key {key} in {split} metric {metric}")
                            val_log_dict[key] = meter_values[meter_known_return]

                    logger.info(f"{split} metrics at epoch {self.epoch} (step {self.steps_taken}):")
                    for key in val_log_dict.keys():
                        logger.info(f"\t~ {key}: {val_log_dict[ key] :.4f}")

                val_log_dict = self._format_logs_for_wandb(val_log_dict, prefix=f"{split}/")
                wandb.log(val_log_dict, step=self.steps_taken)

                # Log individual losses, class losses, and loss histograms
                if is_dataset_indexed:
                    self.log_losses_on_split(split, all_losses)

                # TODO(juan43ramirez): Remove this unnecessary double-sync
                # Check if all the training samples are feasible
                if split == "train" and "violation" in split_metrics:
                    violation_meter = split_metrics["violation"].meters["violation"]
                    if "max" in violation_meter.known_returns:
                        max_train_violation = violation_meter.get_result_dict()["max"]
                    if max_train_violation <= 0:
                        logger.info(f"All training samples are feasible at step {self.steps_taken}!!!")
                        is_all_train_feasible = True

                # Must communicate this to all processes so they can also stop.
                is_all_train_feasible = utils.distributed.do_broadcast_object(is_all_train_feasible)

            logger.info(f"Finished measuring {split} metrics")

        logger.info(f"Finished measuring evaluation metrics")

        utils.distributed.wait_for_all_processes()

        return is_all_train_feasible

    def train(self):
        if self.is_wandb_run_resumed or self.config.checkpointing.is_resuming_from_checkpoint:
            # Retrieves self.{steps_taken, epoch, elapsed_time} and loads checkpointed
            # state_dicts for the model, optimizers and schedulers.
            self._load_checkpoint()
        else:
            self.steps_taken = 0
            self.epoch = 0
            self.elapsed_time = 0
            self.start_time = time.time()
            logger.info("No checkpoint found, starting from scratch.")

            self._save_checkpoint()

        steps_since_last_epoch = self.steps_taken % len(self.dataloaders.train)
        if self.config.data.dataloader.use_distributed_sampler:
            self.dataloaders.train.sampler.set_epoch(self.epoch)

        # Skip the training dataloader ahead to the current step
        train_data_iter = iter(self.dataloaders.train)
        for _ in range(steps_since_last_epoch):
            batch_data = next(train_data_iter)

        for scheduler in [self.schedulers.primal, self.schedulers.dual]:
            if scheduler is not None:
                epochs_taken = int(self.steps_taken // self.num_steps)
                if epochs_taken > 0:
                    logger.info(f"Fast-forwarding {scheduler} to epoch {epochs_taken}")
                for _ in range(epochs_taken):
                    scheduler.step()

        # After loading a checkpoint, and forwarding the dataloader and schedulers,
        # we are ready to train.
        self._train_loop(train_data_iter)

        self.elapsed_time = self.elapsed_time + (time.time() - self.start_time)
        logger.info(f"Completed {self.steps_taken} steps of training" + f" ({self.elapsed_time:.2f} seconds)")

        # Wait for all processes to finish before final validation
        utils.distributed.wait_for_all_processes()

        # Final eval after training if we didn't just do one
        if not self.steps_taken % self.eval_period_steps == 0:
            logger.info("Final model evaluation")
            self._eval_loop()
            self._save_checkpoint()

        logger.info("Training complete")

    def _train_loop(self, train_data_iter):
        logger.info("Starting training")
        self.model.train()

        logger.info(f"Evaluating model performance at step {self.steps_taken}")
        self._eval_loop()
        self.model.train()
        self._save_checkpoint()
        logger.info(f"Evaluation loop completed after step {self.steps_taken}")

        while True:
            try:
                batch_data = next(train_data_iter)
            except StopIteration:
                logger.info(f"Finished epoch {self.epoch}")
                self.epoch += 1
                
                for scheduler in [self.schedulers.primal, self.schedulers.dual]:
                    if scheduler is not None:
                        scheduler.step()
                        # log the learning rate
                        wandb.log({"lr": scheduler.get_last_lr()[0]}, step=self.steps_taken)

                if self.config.data.dataloader.use_distributed_sampler:
                    self.dataloaders.train.sampler.set_epoch(self.epoch)
                train_data_iter = iter(self.dataloaders.train)
                batch_data = next(train_data_iter)

            inputs = batch_data[0].to(device=self.device, non_blocking=True)
            targets = batch_data[1].to(device=self.device, non_blocking=True)
            if isinstance(self.dataloaders.train.dataset, datasets.IndexedDataset):
                # If the training dataset is an IndexedDataset, the third element of the
                # batch_data tuple is the data_indices corresponding to the batch samples.
                data_indices = batch_data[2].to(device=self.device, non_blocking=True)
            else:
                data_indices = None
            constraint_features = data_indices

            compute_cmp_state_fn = lambda: self.cmp.compute_cmp_state(
                model=self.model, inputs=inputs, targets=targets, constraint_features=constraint_features
            )
            cmp_state, lagrangian_store = self.cooper_optimizer.roll(
                compute_cmp_state_fn=compute_cmp_state_fn, return_multipliers=True
            )
            lagrangian, observed_multipliers = lagrangian_store.lagrangian, lagrangian_store.observed_multipliers

            # We syncronize the multipliers across DDP processes after every step.
            self._sync_feasibility_multiplier(observed_multipliers, data_indices)

            if self.is_main_process:
                # TODO(gallego-posada): This is not being aggregated across processes for DDP
                batch_metrics = {key: cmp_state.misc[key] for key in self.config.metrics.batch_level}
                train_log_dict = self._format_logs_for_wandb(batch_metrics, prefix="batch/")
                wandb.log(train_log_dict, step=self.steps_taken)

                if self.cmp.has_dual_variables:
                    max_batch_lambda = observed_multipliers[0].max().detach()
                    wandb.log({"batch/max_lambda": max_batch_lambda}, step=self.steps_taken)

            if self.steps_taken % self.config.logging.print_train_stats_period_steps == 0:
                logger.info(
                    f"Step {self.steps_taken}/{self.num_steps} | Epoch {self.epoch} | Lagrangian: {lagrangian:.4f}"
                )

            self.steps_taken += 1

            if self.steps_taken % self.eval_period_steps == 0:
                logger.info(f"Evaluating model performance at step {self.steps_taken}")
                is_all_train_feasible = self._eval_loop()
                self.model.train()
                self._save_checkpoint()
                logger.info(f"Evaluation loop completed after step {self.steps_taken}")

                if is_all_train_feasible and getattr(self.config.task, "early_stop_on_feasible", False):
                    logger.info("All training datapoints are feasible. Terminating training early.")
                    break

            # Wait for all processes to finish
            utils.distributed.wait_for_all_processes()

            # Check termination criteria
            if self.steps_taken >= self.num_steps:
                break

    def _sync_feasibility_multiplier(self, observed_multipliers, data_indices):
        if self.config.resources.use_ddp and self.dist.multi_gpu and self.cmp.has_dual_variables:
            assert len(observed_multipliers) == 1, "Assumes exactly one constraint group corresponding to feasibility"
            all_worker_multipliers = utils.distributed.do_all_gather_object(observed_multipliers[0])

            all_worker_data_indices = utils.distributed.do_all_gather_object(data_indices)
            for new_multipliers, data_indices in zip(all_worker_multipliers, all_worker_data_indices):
                self.cmp.feasibility_constraint.multiplier.weight.data[data_indices, 0] = new_multipliers

            utils.distributed.wait_for_all_processes()

            if self.is_main_process:
                max_batch_lambda = max([_.max().detach() for _ in all_worker_multipliers])
                wandb.log({"batch/max_lambda": max_batch_lambda}, step=self.steps_taken)

    def generate_reports(self):
        if self.is_main_process:
            config_path = os.path.join(self.run_checkpoint_dir, "config.pkl")
            with open(config_path, "wb") as f:
                pickle.dump(self.config, f)
            wandb.save(config_path, base_path=self.run_checkpoint_dir)

            logger.info(f"Saved checkpoint to {self.run_checkpoint_dir} (step={self.steps_taken}; epoch={self.epoch})")

    def __submitit_checkpoint__(self):
        """Function used by submitit when SLURM job is preempted"""

        self._save_checkpoint()

        resume_config = self.config.copy()
        with resume_config.unlocked():
            resume_config.checkpointing.is_resuming_from_checkpoint = True
        resume_trainer = Trainer(resume_config)
        return submitit.helpers.DelayedSubmission(resume_trainer)
