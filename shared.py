import logging
import os

import rich.logging
import rich.text

MAIN_LOGGER_NAME = "TRAINER"

REGEX_PATTERN = r"((?:\w+\.)*\w+)=([-]?\d+(?:\.\d+)?(?:e[-+]?\d+)?|\w+|\([^)]*\))"


def fetch_main_logger(apply_basic_config=False):
    logger = logging.getLogger(MAIN_LOGGER_NAME)
    if apply_basic_config:
        configure_logger(logger)
    return logger


def configure_logger(logger, custom_format=None, level=logging.INFO, propagate=False, show_path=False):
    logger.propagate = propagate

    for handler in logger.handlers:
        logger.removeHandler(handler)

    format = f"%(module)s:%(funcName)s:%(lineno)d | %(message)s" if custom_format is None else custom_format
    log_formatter = logging.Formatter(format)

    rich_handler = rich.logging.RichHandler(
        markup=True,
        rich_tracebacks=True,
        omit_repeated_times=True,
        show_path=False,
        log_time_format=lambda dt: rich.text.Text.from_markup(f"[red]{dt.strftime('%y-%m-%d %H:%M:%S.%f')[:-4]}"),
    )
    rich_handler.setFormatter(log_formatter)
    logger.addHandler(rich_handler)

    logger.setLevel(level)


def drill_to_key_and_set(_dict, key, value) -> None:
    # Need to split the key by "." and traverse the config to set the new value
    split_key = key.split(".")
    entry_in_config = _dict
    for subkey in split_key[:-1]:
        entry_in_config = entry_in_config[subkey]
    entry_in_config[split_key[-1]] = value


def update_config_with_cli_args(config, variables):
    for key, value in variables.items():
        try:
            value = eval(value)
        except NameError:
            pass
        drill_to_key_and_set(config, key=key, value=value)
