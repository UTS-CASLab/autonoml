# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 21:15:47 2023

@author: David J. Kedziora
"""

import asyncio
import threading
import weakref

# Set up a forever-running asyncio event loop in a thread dedicated to AutonoML.
# This works for Python where there is no pre-existing loop.
# This also works for IPython, which already runs a main-thread pre-existing loop.

loop_autonoml = asyncio.new_event_loop()

def run_loop(in_loop):
    asyncio.set_event_loop(in_loop)
    in_loop.run_forever()

threading.Thread(target=lambda: run_loop(loop_autonoml)).start()

# async def entrypoint():
#     while True:
#         asyncio.sleep(in_duration)

# async def loop_entrypoint():
#     while True:
#         asyncio.sleep(in_duration)

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
    # task = asyncio.create_task(weakly_bound_method())

    global loop_autonoml
    task = asyncio.run_coroutine_threadsafe(coro = weakly_bound_method(),
                                            loop = loop_autonoml)

    # Establishes the task to cancel.
    canceller.task = task

    return task

async def user_pause(in_duration):
    """
    Helper function to mimic an asynchronous pause in user interactions.
    """
    print("USER: Pauses for %i+ seconds." % in_duration)
    await asyncio.sleep(in_duration)