"""Environment detection so the same code/config works on a local laptop and in Colab."""

import importlib.util
import os


def is_colab() -> bool:
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def current__env() -> str:
    """Returns 'colab' or 'local'. Override with the RUN_ENV environment variable."""
    override = os.environ.get("RUN_ENV")
    if override in ("colab", "local"):
        return override
    return "colab" if is_colab() else "local"
