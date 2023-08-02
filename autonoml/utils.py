# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:23:02 2023

@author: David J. Kedziora
"""

import logging
import time
import asyncio
import threading
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


# async def loop_entrypoint():
#     while True:
#         asyncio.sleep(in_duration)

# loop = asyncio.get_event_loop()
#         if loop.is_running() == False:
#             log.debug(("No asyncio event loop is currently running.\n"
#                        "One will be launched for AutonoML operations."))
#             thread_asyncio = threading.Thread(target=self.run_asyncio)
#             thread_asyncio.start()

# loop = asyncio.get_event_loop()
#         if loop.is_running() == False:
#             log.debug(("No asyncio event loop is currently running.\n"
#                        "One will be launched for AutonoML operations."))
#             thread_asyncio = threading.Thread(target=self.run_asyncio).start()
#             # asyncio.run(self.gather_ops())
#             #HOW TO ADD TASKS TO IT?
#         else:
#             log.debug(("The Python environment is already running an asyncio event loop.\n"
#                        "It will be used for AutonoML operations."))
#             asyncio_task_from_method(self.gather_ops)

# def asyncio_task(in_coroutine):
#     loop = asyncio.get_event_loop()
#     if loop.is_running() == False:
#         log.debug(("No asyncio event loop is currently running.\n"
#                     "One will be launched for AutonoML operations."))
#         thread_asyncio = threading.Thread(target=loop_entrypoint).start()
#     else:
#         task = asyncio.create_task(in_coroutine)
#     return task

# def asyncio_task_from_method(in_method):
#     """ 
#     Ensures proper handling of asynchronous tasks formed from bound methods.
#     Specifically, weakly references the binding instance, i.e. self.
#     Once self drops out of scope, the task should cancel.
#     """
#     class Canceller:
#         def __call__(self, in_proxy):
#             self.task.cancel()

#     canceller = Canceller()

#     # The canceller is called when the proxy is finalised.
#     proxy_object = weakref.proxy(in_method.__self__, canceller)

#     weakly_bound_method = in_method.__func__.__get__(proxy_object)
#     task = asyncio.create_task(weakly_bound_method())

#     # Establishes the task to cancel.
#     canceller.task = task

#     return task

# async def user_pause(in_duration):
#     """
#     Helper function to mimic an asynchronous pause in user interactions.
#     """
#     print("USER: Pauses for %i+ seconds." % in_duration)
#     await asyncio.sleep(in_duration)