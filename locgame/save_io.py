import ml_utils.save_io as io
from locgame.models import *

def load_model(path, load_sd=True, **kwargs):
    """
    Loads the model architecture and state dict from a .pt or .pth
    file. Or from a training save folder. Defaults to the last check
    point file saved in the save folder.

    path: str
        either .pt,.p, or .pth checkpoint file; or path to save folder
        that contains multiple checkpoints
    models: dict
        this is easiest if you simply pass `globals()` as this parameter

        keys: str
            the class names of the potential models
        vals: Class
            the potential model classes
    load_sd: bool
        if true, the saved state dict is loaded. Otherwise only the
        model architecture is loaded with a random initialization.
    """
    return io.load_model(path, models=globals(), load_sd=load_sd,
                                                 verbose=True)
