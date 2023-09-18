# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:23:02 2023

@author: David J. Kedziora
"""

import __main__
import logging
import sys
import os
import time
import traceback

from typing import Type

# Explicitly grab a logging handle for the AutonoML codebase.
# Setup is done during package initialisation.
log = logging.getLogger("autonoml")
log_level = logging.DEBUG

def setup_logger(in_filename_script: str):
    global log
    global log_level

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
        if in_filename_script:
            log_file = os.path.splitext(in_filename_script)[0] + ".log"
            handler_file = logging.FileHandler(log_file, mode = "w")
            handler_file.setLevel(log_level)
            log.addHandler(handler_file)

setup_logger(in_filename_script = getattr(__main__, "__file__", None))

class Timestamp:
    """
    A wrapper for timestamps.
    
    If initialised with arbitrary argument, is 'fake', e.g. Timestamp(None).
    Fake Timestamps are printed out as blank spaces, i.e. an indent.
    """
    
    def __init__(self, is_real: bool = True):
        self.time = None
        self.ms = None
        if is_real:
            self.time = time.time()
            self.ms = repr(self.time).split(".")[1][:3]
        
    def __str__(self):
        if self.time:
            return time.strftime("%y-%m-%d %H:%M:%S.{}".format(self.ms), 
                                 time.localtime(self.time))
        else:
            return " "*21
    
    def update_from(self, in_timestamp: Type["Timestamp"]):
        self.time = in_timestamp.time
        self.ms = repr(self.time).split(".")[1][:3]

class CustomBool:
    def __init__(self, value):
        value = str(value).lower()
        if value in ("0", "false", "n", "no"):
            self.value = False
        elif value in ("1", "true", "y", "yes"):
            self.value = True
        else:
            raise ValueError("Invalid value for CustomBool.")

    def __bool__(self):
        return self.value

    def __repr__(self):
        return "y" if self.value else "n"
    
def identify_error(in_exception: Exception, in_text_alert: str):
    log.error(in_text_alert)
    log.debug("Exception: %s" % str(in_exception))
    log.debug("Traceback: %s" % "".join(traceback.format_tb(in_exception.__traceback__)))