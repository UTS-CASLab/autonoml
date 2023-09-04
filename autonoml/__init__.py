"""
AutonoML root module.
"""

import sys
import logging

import os
import __main__

#%% Set up a module-level logger.
log = logging.getLogger("autonoml")

log_level = logging.DEBUG

# Only construct logger handlers the first time the module is loaded.
if not log.handlers:
    # The debug handler catches all messages below 'info' priority.
    # It sends them to stdout with a detailed format.
    handler_debug = logging.StreamHandler(sys.stdout)
    handler_debug.setLevel(0)
    handler_debug.addFilter(type("ThresholdFilter", (object,), {"filter": lambda x, logRecord: logRecord.levelno < logging.INFO})())
    formatter_debug = logging.Formatter("%(levelname)s {%(filename)s:%(lineno)d} - %(message)s")
    handler_debug.setFormatter(formatter_debug)
    
    # The info handler catches all messages at 'info' priority.
    # It sends them to stdout.
    handler_info = logging.StreamHandler(sys.stdout)
    handler_info.setLevel(logging.INFO)
    handler_info.addFilter(type("ThresholdFilter", (object,), {"filter": lambda x, logRecord: logRecord.levelno < logging.WARNING})())
    
    # The warning handler catches all messages at 'warning' priority or above.
    # It sends them to stderr.
    handler_warning = logging.StreamHandler(sys.stderr)
    handler_warning.setLevel(logging.WARNING)
    
    # Attach handlers to the logger and specify a minimum level of attention.
    log.addHandler(handler_debug)
    log.addHandler(handler_info)
    log.addHandler(handler_warning)
    log.setLevel(log_level)

    # Create a file handler if a Python script imported this package.
    calling_script = __main__.__file__
    if calling_script:
        log_file = os.path.splitext(calling_script)[0] + ".log"
        handler_file = logging.FileHandler(log_file)
        handler_file.setLevel(log_level)  # Adjust the log level as needed
        log.addHandler(handler_file)

#%% Ensure all required functions are available upon importing the root module.

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
from .utils import *
from .concurrency import *
from .strategy import *