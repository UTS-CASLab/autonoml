# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:23:02 2023

@author: David J. Kedziora
"""

import logging
import time
import asyncio

# Explicitly grab a handle for the AutonoML codebase.
# Detailed specifications are set in the __init__.py file.
log = logging.getLogger("autonoml")


async def user_pause(in_duration):
    """
    Helper function to mimic an asynchronous pause in user interactions.
    """
    print("USER: Pauses for %i+ seconds." % in_duration)
    await asyncio.sleep(in_duration)


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