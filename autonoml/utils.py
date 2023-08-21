# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:23:02 2023

@author: David J. Kedziora
"""

import logging
import time

# Explicitly grab a handle for the AutonoML codebase.
# Detailed specifications are set in the __init__.py file.
log = logging.getLogger("autonoml")

class Timestamp:
    """
    A wrapper for timestamps.
    
    If initialised with arbitrary argument, is 'fake', e.g. Timestamp(None).
    Fake Timestamps are printed out as blank spaces, i.e. an indent.
    """
    
    def __init__(self, is_real = True):
        self.time = None
        self.ms = None
        if is_real:
            self.time = time.time()
            self.ms = repr(self.time).split('.')[1][:3]
        
    def __str__(self):
        if self.time:
            return time.strftime("%y-%m-%d %H:%M:%S.{}".format(self.ms), 
                                 time.localtime(self.time))
        else:
            return " "*21
    
    def update_from(self, in_timestamp):
        self.time = in_timestamp.time
        self.ms = repr(self.time).split('.')[1][:3]