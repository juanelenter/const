import abc
from types import SimpleNamespace

import cooper
import torch

import shared
from src.utils.metrics import cross_entropy, l2_loss, top1_accuracy

logger = shared.fetch_main_logger()


def forward_and_loss_helper(model, inputs, targets, pointwise_loss_level, apply_clamp=False, regression=False):
    logits = model(inputs)

    if not regression:
        per_sample_loss = cross_entropy(logits, targets, per_sample=True)
        per_sample_acc = top1_accuracy(logits, targets, per_sample=True)
        average_acc = per_sample_acc.sum() / logits.shape[0]
    else:
        per_sample_loss = l2_loss(logits, targets, per_sample=True)

    if apply_clamp:
        per_sample_loss = torch.clamp(per_sample_loss, min=pointwise_loss_level)

    average_loss = per_sample_loss.sum() / logits.shape[0]

    if not regression:
        return per_sample_loss, per_sample_acc, average_loss, average_acc
    else:
        return per_sample_loss, average_loss


class BaseProblem(cooper.ConstrainedMinimizationProblem, abc.ABC):
    has_dual_variables: bool

    @abc.abstractmethod
    def compute_cmp_state(self) -> cooper.CMPState:
        pass

    @abc.abstractmethod
    def pointwise_loss_level(self):
        pass

    @abc.abstractmethod
    def dual_parameter_groups(self):
        pass

    @abc.abstractmethod
    def extract_multiplier_stats(self):
        pass

    def compute_excess_loss(self, per_sample_loss):
        return per_sample_loss - self.pointwise_loss_level()


class ERMClassificationProblem(BaseProblem):
    has_dual_variables = False

    def __init__(self, target_pointwise_loss: float):
        self.pointwise_loss_clamp = target_pointwise_loss
        logger.info(f"{self.__class__.__name__} will clamp pointwise losses below {self.pointwise_loss_clamp}")

        super().__init__()

    def loss_fn(self, *args, **kwargs):
        return cross_entropy(*args, **kwargs)

    def pointwise_loss_level(self):
        return self.pointwise_loss_clamp

    def dual_parameter_groups(self):
        return None

    def compute_cmp_state(self, model, inputs, targets, constraint_features=None) -> cooper.CMPState:
        # We set apply_clamp=True in all cases since when no `target_pointwise_loss` is
        # provided in the config, it is set to zero.
        # Since CrossEntropy and L2 losses are greater than or equal to zero, the clamp
        # becomes a no-op.
        per_sample_loss, per_sample_acc, average_loss, average_acc = forward_and_loss_helper(
            model, inputs, targets, self.pointwise_loss_clamp, apply_clamp=True, regression=False
        )

        batch_log_metrics = dict(
            avg_loss=average_loss.detach(),
            avg_acc=average_acc,
            max_loss=per_sample_loss.max().detach(),
        )
        return cooper.CMPState(loss=average_loss, observed_constraints=[], misc=batch_log_metrics)

    def extract_multiplier_stats(self):
        return None

class WeightedERMClassificationProblem(BaseProblem):
    has_dual_variables = False
    has_class_weights = True

    def __init__(
        self,
        target_pointwise_loss: float,
        class_weights: SimpleNamespace,
        dataset_metadata: SimpleNamespace,
        device: torch.device,
    ):
        self.pointwise_loss_clamp = target_pointwise_loss
        self.class_weights = self._create_class_weights(class_weights, dataset_metadata, device)
        logger.info(f"{self.__class__.__name__} will clamp pointwise losses below {self.pointwise_loss_clamp}")
        super().__init__()

    def loss_fn(self, *args, **kwargs):
        return cross_entropy(*args, **kwargs)

    def pointwise_loss_level(self):
        return self.pointwise_loss_clamp

    def dual_parameter_groups(self):
        return None

    def compute_cmp_state(self, model, inputs, targets, constraint_features=None) -> cooper.CMPState:
        # We set apply_clamp=True in all cases since when no `target_pointwise_loss` is
        # provided in the config, it is set to zero.
        # Since CrossEntropy and L2 losses are greater than or equal to zero, the clamp
        # becomes a no-op.
        per_sample_loss, per_sample_acc, average_loss, average_acc = forward_and_loss_helper(
            model, inputs, targets, self.pointwise_loss_clamp, apply_clamp=True, regression=False
        )

        batch_log_metrics = dict(
            avg_loss=average_loss.detach(),
            avg_acc=average_acc,
            max_loss=per_sample_loss.max().detach(),
        )
        weighted_loss = (per_sample_loss * self.class_weights[targets]).mean()

        return cooper.CMPState(loss=weighted_loss, observed_constraints=[], misc=batch_log_metrics)

    def extract_multiplier_stats(self):
        return None

    def _create_class_weights(self, weight_config, dataset_metadata, device, normalize=True) -> torch.Tensor:
        """creates class weights for imbalanced datasets
        effective implements the weighting described in https://arxiv.org/abs/1901.05555
        following its official implementation https://github.com/richardaecn/class-balanced-loss/blob/master/src/cifar_main.py#L425-L430
        """
        if weight_config.weights is None:
            if weight_config.type == "effective":
                num_classes = len(dataset_metadata.imbalance_metadata.samples_per_class)
                num_samples = dataset_metadata.imbalance_metadata.samples_per_class
                class_weights = []
                total = 0
                beta = weight_config.beta
                for i in range(num_classes):
                    w = (1 - beta) / (1 - beta ** num_samples[i])
                    class_weights.append(w)
                    total += w
                class_weights = [r / total for r in class_weights]
            else:
                raise NotImplementedError

        class_weights = torch.tensor(class_weights, device=device)

        return class_weights


class ERMRegressionProblem(BaseProblem):
    has_dual_variables = False

    def __init__(self, target_pointwise_loss: float):
        self.pointwise_loss_clamp = target_pointwise_loss
        logger.info(f"{self.__class__.__name__} will clamp pointwise losses below {self.pointwise_loss_clamp}")

        super().__init__()

    def loss_fn(self, *args, **kwargs):
        return l2_loss(*args, **kwargs)

    def pointwise_loss_level(self):
        return self.pointwise_loss_clamp

    def dual_parameter_groups(self):
        return None

    def compute_cmp_state(self, model, inputs, targets, constraint_features=None) -> cooper.CMPState:
        # We set apply_clamp=True in all cases since when no `target_pointwise_loss` is
        # provided in the config, it is set to zero.
        # Since CrossEntropy and L2 losses are greater than or equal to zero, the clamp
        # becomes a no-op.
        # breakpoint()
        per_sample_loss, average_loss = forward_and_loss_helper(
            model, inputs, targets, self.pointwise_loss_clamp, apply_clamp=True, regression=True
        )

        batch_log_metrics = dict(
            avg_loss=average_loss.detach(),
            max_loss=per_sample_loss.max().detach(),
        )
        return cooper.CMPState(loss=average_loss, observed_constraints=[], misc=batch_log_metrics)

    def extract_multiplier_stats(self):
        return None


class FeasibleClassificationProblem(BaseProblem):
    has_dual_variables = True

    def __init__(
        self, target_pointwise_loss: float, use_strict_accuracy: bool, multiplier: cooper.multipliers.Multiplier
    ):
        self.use_strict_accuracy = use_strict_accuracy
        self.target_pointwise_loss = target_pointwise_loss

        self.feasibility_constraint = cooper.ConstraintGroup(
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.FormulationType.LAGRANGIAN,
            multiplier=multiplier,
        )
        self.constraint_groups = [self.feasibility_constraint]

        super().__init__()

    def loss_fn(self, *args, **kwargs):
        return cross_entropy(*args, **kwargs)

    def pointwise_loss_level(self):
        return self.target_pointwise_loss

    def dual_parameter_groups(self):
        return {"multipliers": self.feasibility_constraint.multiplier.parameters()}

    def compute_cmp_state(self, model, inputs, targets, constraint_features) -> cooper.CMPState:
        per_sample_loss, per_sample_acc, average_loss, average_acc = forward_and_loss_helper(
            model, inputs, targets, self.target_pointwise_loss, apply_clamp=False, regression=False
        )

        violation = self.compute_excess_loss(per_sample_loss)

        # This is a "greater-than" constraint: accuracy >= 1.0
        # So in "less-than" convention, we have - accuracy + 1.0 <= 0
        # Use 0.5 as the threshold for strict accuracy
        # TODO(gallego-posada): Document this bettter
        strict_violation = -per_sample_acc + 0.5 if self.use_strict_accuracy else None

        constraint_state = cooper.ConstraintState(
            violation=violation, strict_violation=strict_violation, constraint_features=constraint_features
        )
        observed_constraints = [(self.feasibility_constraint, constraint_state)]

        batch_log_metrics = dict(
            avg_loss=average_loss.detach(),
            avg_acc=average_acc,
            max_loss=per_sample_loss.max().detach(),
        )
        return cooper.CMPState(loss=None, observed_constraints=observed_constraints, misc=batch_log_metrics)

    def evaluate_multipliers(self, constraint_features=None):
        if constraint_features is None:
            return self.feasibility_constraint.multiplier()
        else:
            return self.feasibility_constraint.multiplier(constraint_features)

    def extract_multiplier_stats(self):
        if not isinstance(self.feasibility_constraint.multiplier, cooper.multipliers.ExplicitMultiplier):
            raise NotImplementedError("This function is only intended to be used with `ExplicitMultiplier`s")

        all_multiplier_values = self.feasibility_constraint.multiplier.weight.data.detach()
        multiplier_stats = {
            "max": all_multiplier_values.max(),
            "avg": all_multiplier_values.mean(),
            "median": all_multiplier_values.median(),
            "rate_zeros": (all_multiplier_values == 0).float().mean(),
            "all_multiplier_values": all_multiplier_values,
            # TODO(juan43ramirez): log a few quantiles of the multiplier values
        }

        return multiplier_stats


class FeasibleRegressionProblem(BaseProblem):
    has_dual_variables = True

    def __init__(
        self, target_pointwise_loss: float, use_strict_accuracy: bool, multiplier: cooper.multipliers.Multiplier
    ):
        self.use_strict_accuracy = use_strict_accuracy
        self.target_pointwise_loss = target_pointwise_loss

        self.feasibility_constraint = cooper.ConstraintGroup(
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.FormulationType.LAGRANGIAN,
            multiplier=multiplier,
        )
        self.constraint_groups = [self.feasibility_constraint]

        super().__init__()

    def loss_fn(self, *args, **kwargs):
        return l2_loss(*args, **kwargs)

    def pointwise_loss_level(self):
        return self.target_pointwise_loss

    def dual_parameter_groups(self):
        return {"multipliers": self.feasibility_constraint.multiplier.parameters()}

    def compute_cmp_state(self, model, inputs, targets, constraint_features) -> cooper.CMPState:
        per_sample_loss, average_loss = forward_and_loss_helper(
            model, inputs, targets, self.target_pointwise_loss, apply_clamp=False, regression=True
        )

        violation = self.compute_excess_loss(per_sample_loss)

        # This is a "greater-than" constraint: accuracy >= 1.0
        # So in "less-than" convention, we have - accuracy + 1.0 <= 0
        # Use 0.5 as the threshold for strict accuracy
        # TODO(gallego-posada): Document this bettter
        # strict_violation = -per_sample_acc + 0.5 if self.use_strict_accuracy else None

        constraint_state = cooper.ConstraintState(violation=violation, constraint_features=constraint_features)
        observed_constraints = [(self.feasibility_constraint, constraint_state)]

        batch_log_metrics = dict(
            avg_loss=average_loss.detach(),
            max_loss=per_sample_loss.max().detach(),
        )
        return cooper.CMPState(loss=None, observed_constraints=observed_constraints, misc=batch_log_metrics)

    def evaluate_multipliers(self, constraint_features=None):
        if constraint_features is None:
            return self.feasibility_constraint.multiplier()
        else:
            return self.feasibility_constraint.multiplier(constraint_features)

    def extract_multiplier_stats(self):
        if not isinstance(self.feasibility_constraint.multiplier, cooper.multipliers.ExplicitMultiplier):
            raise NotImplementedError("This function is only intended to be used with `ExplicitMultiplier`s")

        all_multiplier_values = self.feasibility_constraint.multiplier.weight.data.detach()
        multiplier_stats = {
            "lambda/max": all_multiplier_values.max(),
            "lambda/avg": all_multiplier_values.mean(),
            "lambda/median": all_multiplier_values.median(),
            "lambda/rate_zeros": (all_multiplier_values == 0).float().mean(),
            "all_multiplier_values": all_multiplier_values,
        }

        return multiplier_stats
