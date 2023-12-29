from dataclasses import dataclass

import ml_collections as mlc

MLC_PH = mlc.config_dict.config_dict.placeholder


@dataclass
class OptimizerConfig:
    name: str = MLC_PH(str)
    per_group_kwargs: mlc.ConfigDict = mlc.ConfigDict()
    shared_kwargs: mlc.ConfigDict = mlc.ConfigDict()
    scheduler: mlc.ConfigDict = mlc.ConfigDict({"name": MLC_PH(str), "kwargs": MLC_PH(dict)})

    def todict(self):
        """Wraps an ML-Collections config dict around the dataclass."""

        config = mlc.ConfigDict()

        config.name = self.name

        config.per_group_kwargs = mlc.ConfigDict()
        config.per_group_kwargs = self.per_group_kwargs
        config.shared_kwargs = mlc.ConfigDict()
        config.shared_kwargs = self.shared_kwargs

        config.scheduler = mlc.ConfigDict()
        config.scheduler.name = MLC_PH(str)
        config.scheduler.kwargs = mlc.ConfigDict()
        if getattr(self.scheduler, "name", None) is not None:
            config.scheduler.name = self.scheduler["name"]
        if getattr(self.scheduler, "kwargs", None) is not None:
            config.scheduler.kwargs = self.scheduler["kwargs"]

        return config
