import warnings
warnings.filterwarnings("ignore", message="Tight layout not applied.*", module="matplotlib")

from . import main as _main
from . import parallel as _parallel
from . import functions as _func
from joblib import Parallel as Parallel, delayed as delayed

_modules = [_main, _parallel, _func]

for _mod in _modules:
    for _name in dir(_mod):
        if not _name.startswith("__"):
            globals()[_name] = getattr(_mod, _name)

def __getattr__(name):
    for mod in _modules:
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(name)

def __setattr__(name, value):
    for mod in _modules:
        if hasattr(mod, name):
            setattr(mod, name, value)
    globals()[name] = value

__all__ = [name for name in globals() if not name.startswith("__")]

# Expose submodules
main = _main
parallel = _parallel
functions = _func

# Use the version of run_pipeline_parallel from :mod:`main` by default
run_pipeline_parallel = _main.run_pipeline_parallel

