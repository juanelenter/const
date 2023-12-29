import abc
import inspect

import torch
import torch.distributed

import src.utils.distributed as dist_utils


def skewness(t):
    """
    Computes the skewness of a tensor.
    """

    return torch.mean(((t - t.mean()) / t.std()) ** 3)


def kurtosis(t):
    """
    Computes the kurtosis of a tensor.
    """

    return torch.mean(((t - t.mean()) / t.std()) ** 4)


def inter_qantile_range(t):
    """
    Computes the iqr of a tensor.
    """

    return torch.quantile(t, 0.75) - torch.quantile(t, 0.25)


class Meter(abc.ABC):
    """
    Abstract class specifying the Meters interface
    """

    def __init__(self):
        self.reset()
        self.known_returns = None

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, val, n=1):
        pass

    @abc.abstractmethod
    def _sync(self):
        pass

    @abc.abstractmethod
    def get_result_dict(self):
        pass

    @classmethod
    def args_for_update(cls):
        update_args = inspect.getfullargspec(cls.update).args
        update_args.remove("self")
        return update_args


class BestMeter(Meter):
    """Computes and stores the best observed value of a metric."""

    def __init__(self, direction="max"):
        assert direction in {"max", "min"}
        self.direction = direction
        self.known_returns = [self.direction]
        self.reset()

    def reset(self):
        self.val = -float("inf") if self.direction == "max" else float("inf")

    def _sync(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered_values = dist_utils.do_all_gather_object(self.val)
            value_for_update = max(gathered_values) if self.direction == "max" else min(gathered_values)
            self.update(val=value_for_update)

    def update(self, val) -> bool:
        """Update meter and return boolean flag indicating if the current value is
        the best so far."""

        best_incoming = max(val) if self.direction == "max" else min(val)

        if self.direction == "max":
            if best_incoming > self.val:
                self.val = best_incoming
                return True
        elif self.direction == "min":
            if best_incoming < self.val:
                self.val = best_incoming
                return True

        return False

    def get_result_dict(self):
        self._sync()
        return {self.direction: self.val}


class AverageMeter(Meter):
    """Computes and stores the average and current value of a given metric. Supports
    distributed training.
    """

    def __init__(self):
        self.known_returns = ["avg"]
        self.reset()

    def reset(self):
        self.synced_count = 0
        self.new_count = 0

        self.synced_sum = 0
        self.new_sum = 0

    def update(self, val, n=1):
        self.new_sum += val * n
        self.new_count += n

    def _sync(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            new_counts = sum(dist_utils.do_all_gather_object(self.new_count))
            new_sums = sum(dist_utils.do_all_gather_object(self.new_sum))
        else:
            new_counts, new_sums = self.new_count, self.new_sum

        self.synced_count += new_counts
        self.synced_sum += new_sums

        self.new_count = 0
        self.new_sum = 0

    @property
    def avg(self):
        self._sync()
        # No need to add new_count since call to avg() forces a call to _sync()
        if self.synced_count > 0:
            return self.synced_sum / self.synced_count
        else:
            return 0

    def get_result_dict(self):
        return {"avg": self.avg}


class StatsMeter(Meter):
    """Stores all values in order to compute statistics of a given metric."""

    STATS_MAP = {
        "min": torch.min,
        "median": torch.median,
        "avg": torch.mean,
        "max": torch.max,
        "std": torch.std,
        "nonpos_rate": lambda x: (x <= 0).float().mean(),
        "pos_avg": lambda x: x[x > 0].mean(),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "iqr": inter_qantile_range,
        "pointwise": lambda x: x,
    }

    def __init__(self, stats=["min", "median", "max", "std", "pointwise"]):
        self.known_returns = stats
        self.reset()

    def reset(self):
        self.all_elements = []

    def update(self, val):
        self.all_elements += [val.detach()]

    def _sync(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            cat_all_seen_elements = torch.cat(self.all_elements)
            self.all_elements = dist_utils.do_all_gather_object(cat_all_seen_elements)
        return torch.cat(self.all_elements)

    def get_result_dict(self):
        cat_all_elements = self._sync()
        return {stat: self.STATS_MAP[stat](cat_all_elements) for stat in self.known_returns}
