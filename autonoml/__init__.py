"""
AutonoML root module.
"""

#%% Utilities must be imported first as this sets up a module-wide logger.

from .utils import *

#%% Ensure all other required functions are available upon importing the root module.

# IMPORTANT...
# All functions/variables below are exposed to a user and can be used when scripting.
# This codebase leverages multiprocessing, so user scripts must be neutralised in child processes.
# Critically, process spawners should be blocked off or recursion will crash CPUs!
# CPU-intensive functions should also be blocked off.
# However, ideally, it should not be up to the user to use multiprocessing guards.
# Hence, it is up to codebase developers to intercept such calls, within reason.
# TODO: See if this is even possible.

from .core import *
from .streamer import *

from .settings import *
from .concurrency import *
from .strategy import *
from .solver import AllocationMethod

# TODO: Clean up what is accessible to a user.