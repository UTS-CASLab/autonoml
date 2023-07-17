# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:23:02 2023

@author: David J. Kedziora
"""

import logging
import time
import asyncio
import weakref

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



def asyncio_task_from_method(in_method):
    """ 
    Ensures proper handling of asynchronous tasks formed from bound methods.
    Specifically, weakly references the binding instance, i.e. self.
    Once self drops out of scope, the task should cancel.
    """
    class Canceller:
        def __call__(self, in_proxy):
            self.task.cancel()

    canceller = Canceller()

    # The canceller is called when the proxy is finalised.
    proxy_object = weakref.proxy(in_method.__self__, canceller)

    weakly_bound_method = in_method.__func__.__get__(proxy_object)
    task = asyncio.create_task(weakly_bound_method())

    # Establishes the task to cancel.
    canceller.task = task

    return task

async def user_pause(in_duration):
    """
    Helper function to mimic an asynchronous pause in user interactions.
    """
    print("USER: Pauses for %i+ seconds." % in_duration)
    await asyncio.sleep(in_duration)