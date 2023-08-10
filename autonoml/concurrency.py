# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 21:15:47 2023

@author: David J. Kedziora
"""

import asyncio
import threading
import weakref
import functools
import atexit

# Set up a forever-running asyncio event loop in a thread dedicated to AutonoML.
# This works for Python where there is no pre-existing loop.
# This also works for IPython, which already runs a main-thread pre-existing loop.
# TODO: Check whether there are memory leaks once a script ends.

loop_autonoml = asyncio.new_event_loop()

def run_loop(in_loop):
    asyncio.set_event_loop(in_loop)
    try:
        in_loop.run_forever()
    finally:
        # If the loop is stopped, this is triggered.
        in_loop.close()

# Create the thread to run the loop.
thread_autonoml = threading.Thread(target=lambda: run_loop(loop_autonoml))
thread_autonoml.start()

# TODO: Consider cancelling all tasks in the loop.
async def asyncio_stop_loop():
    loop = asyncio.get_event_loop()
    loop.stop()

def end_loop():
    """
    Stops the AutonoML event loop running.
    This triggers it to close, which also stops the associated thread.
    This should only be called when a program using the package is exited.
    """
    global loop_autonoml, thread_autonoml
    asyncio.run_coroutine_threadsafe(coro = asyncio_stop_loop(),
                                     loop = loop_autonoml)

# Trigger the shutdown of the AutonoML end loop when it is no longer needed.
# TODO: Work out if there is a way to test that this actually works.
atexit.register(end_loop)



def inspect_loop():
    """
    A debugging function to see what tasks exist on asyncio event loops.
    It will operate on the calling thread and the AutonoML-dedicated thread.
    These two threads may be the same, depending on where the function is called.
    """

    print("Asynchronous tasks running on the current-thread event loop, "
          "if it exists for the Python implementation...")
    print("Warning: If this function is called within an AutonoML coroutine, "
          "it will inspect the dedicated AutonoML loop.")
    loop = asyncio.get_event_loop()
    try:
        all_tasks = asyncio.all_tasks(loop)
    except:
        all_tasks = [None]
    for task in all_tasks:
        print(task)
    print()

    print("Asynchronous tasks running on the alternate-thread event loop "
          "dedicated to AutonoML...")
    for task in asyncio.all_tasks(loop_autonoml):
        print(task)
    print()



def asyncio_task_from_method(from_sync_thread, in_method, *args, **kwargs):
    """ 
    Ensures proper handling of asynchronous tasks formed from bound methods.
    Specifically, weakly references the binding instance, i.e. self.
    Once self drops out of scope, the task should cancel.

    Notes...
    The return type depends on whether 'from_sync_thread' is True or False.
    See 'create_async_task' and 'create_async_task_from_sync' functions.
    """

    class Canceller:
        def __call__(self, in_proxy):
            self.future.cancel()

    canceller = Canceller()

    # The canceller is called when the proxy is finalised.
    proxy_object = weakref.proxy(in_method.__self__, canceller)
    weakly_bound_method = in_method.__func__.__get__(proxy_object)

    global loop_autonoml
    # print(weakly_bound_method)
    # print(args)
    # print(kwargs)
    if from_sync_thread:
        future = asyncio.run_coroutine_threadsafe(coro = weakly_bound_method(*args, **kwargs),
                                                  loop = loop_autonoml)
    else:
        future = loop_autonoml.create_task(weakly_bound_method(*args, **kwargs))

    # Establishes the future to cancel.
    canceller.future = future

    return future

def create_async_task_from_sync(in_method, *args, **kwargs):
    """
    Creates an asyncio coroutine out of a bound method and puts it on an event loop.
    This function is called from the main synchronous thread in a threadsafe manner.
    Returns a concurrent.futures.Future, which can be awaited as follows.
        try:
            result = task.result(timeout=timeout)
            ...
        except concurrent.futures.TimeoutError:
            ...
        except concurrent.futures.CancelledError:
            ...
    """
    return asyncio_task_from_method(True, in_method, *args, **kwargs)

def create_async_task(in_method, *args, **kwargs):
    """
    Creates an asyncio coroutine out of a bound method and puts it on an event loop.
    This function is called within running coroutines.
    Returns an asyncio.Future, which can be awaited as usual with 'await'.
    """
    return asyncio_task_from_method(False, in_method, *args, **kwargs)



# TODO: Verify the ChatGPT suggestion of binding method to instance. Seems to work.
def schedule_this(bound_method):
    @functools.wraps(bound_method)
    def wrapper_decorator(self, *args, **kwargs):
        bound_method_with_instance = bound_method.__get__(self, self.__class__)
        create_async_task_from_sync(bound_method_with_instance, *args, **kwargs)
    return wrapper_decorator



async def user_pause(in_duration):
    """
    Helper function to mimic an asynchronous pause in user interactions.
    """
    print("USER: Pauses for %i+ seconds." % in_duration)
    await asyncio.sleep(in_duration)