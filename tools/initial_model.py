import importlib
import os
import time
from functools import reduce
import sys
import torch





def initialize_module(path: str, args: dict = None, initialize: bool = True):
    """
    Load module dynamically with "args".
    Args:
        path: module path in this project.
        args: parameters that passes to the Class or the Function in the module.
        initialize: initialize the Class or the Function with args.
    Examples:
        Config items are as follows：
            [model]
            path = "modules.ecapa_tdnn.SpeakerIdetification"
            [model.args]
            ...
        This function will:
            1. Load the "model.full_sub_net" module.
            2. Call "FullSubNetModel" Class (or Function) in "model.full_sub_net" module.
            3. If initialize is True:
                instantiate (or call) the Class (or the Function) and pass the parameters (in "[model.args]") to it.
    """
    module_path = ".".join(path.split(".")[:-1])
    class_or_function_name = path.split(".")[-1]
    module = importlib.import_module(module_path)
    class_or_function = getattr(module, class_or_function_name)

    if initialize:
        if args:
            return class_or_function(**args)
        else:
            return class_or_function()
    else:
        return class_or_function