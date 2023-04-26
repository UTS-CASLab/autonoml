# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:23:02 2023

@author: David J. Kedziora
"""

import logging
import time

log = logging.getLogger("autonoml")

class Timestamp:
    
    def __init__(self):
        self.time = time.time()
        self.ms = repr(self.time).split('.')[1][:3]
        
    def __str__(self):
        return time.strftime("%y-%m-%d %H:%M:%S.{}".format(self.ms), 
                             time.localtime(self.time))
    
    def update_from(self, in_timestamp):
        self.time = in_timestamp.time
        self.ms = repr(self.time).split('.')[1][:3]