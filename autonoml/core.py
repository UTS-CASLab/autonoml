# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:39:37 2023

@author: David J. Kedziora
"""

from .utils import log
from .settings import SystemSettings as SS

import asyncio
from aioconsole import ainput

import time



class Timestamp:
    
    def __init__(self):
        self.time = time.time()
        
    def __str__(self):
        return time.strftime('%y-%m-%d %H:%M:%S', time.localtime(self.time))



class DataSupply:
    """
    A collection of data that supplies machine learning processes.
    """
    
    def __init__(self):
        log.info("%s - A DataSupply has been initialised." % Timestamp())



class AutonoMachine:
    """
    A system designed to autonomously process a machine learning task.
    """
    
    def __init__(self):
        log.info("%s - An AutonoMachine has been initialised." % Timestamp())
        
        self.dataports = dict()
        
        self.is_running = False
        self.delay_until_check = SS.BASE_DELAY_UNTIL_CHECK
        
        self.ops = None
        
        self.run()
        
    def run(self):
        log.info("%s - The AutonoMachine is now running." % Timestamp())
        self.is_running = True
        
        loop = asyncio.get_event_loop()
        if loop.is_running() == False:
            log.debug(("No asyncio event loop is currently running.\n"
                       "One will be launched for AutonoML operations."))
            asyncio.run(self.gather_ops())
        else:
            log.debug(("The Python environment is already running an asyncio event loop.\n"
                       "It will be used for AutonoML operations."))
            loop.create_task(self.gather_ops())
            
    def stop(self):
        log.info("%s - The AutonoMachine is now stopping." % Timestamp())
        self.is_running = False
        
        # Cancel all asynchronous operations.
        if self.ops:
            for op in self.ops:
                op.cancel()
            
    async def gather_ops(self):
        self.ops = [asyncio.create_task(op) for op in [self.check_stop(),
                                                       self.check_issues()]]
        await asyncio.gather(*self.ops, return_exceptions=True)
        
    async def check_stop(self):
        while self.is_running:
            await asyncio.sleep(10)
            self.stop()
        
    async def check_issues(self):
        while self.is_running:
            await asyncio.sleep(self.delay_until_check)
            log.warning("%s - Whoop." % Timestamp())