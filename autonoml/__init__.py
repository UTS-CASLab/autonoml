"""
AutonoML root module.
"""

#%% Utilities must be imported first as this sets up a module-wide logger.

from .utils import *

#%% Ensure all other required functions are available upon importing the root module.

from .core import *
from .streamer import *

from .settings import *
from .concurrency import *
from .strategy import *
from .solution import AllocationMethod

# TODO: Clean up what is accessible to a user.