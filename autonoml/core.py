# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:39:37 2023

@author: David J. Kedziora
"""

from .utils import log

import asyncio
from aioconsole import ainput

import time


class Timestamp:
    
    def __init__(self):
        self.time = time.time()
        
    def __str__(self):
        return time.strftime('%y-%m-%d %H:%M:%S', time.localtime(self.time))

# def get_time():
#     time_now = time.time()
#     return time_now
    
# def get_time_pretty():
#     date_time = datetime.fromtimestamp(timestamp)

# convert timestamp to string in dd-mm-yyyy HH:MM:SS
# str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")


class DataSupply:
    """
    A collection of data that supplies machine learning processes.
    """
    
    def __init__(self):
        log.info("%s - A DataSupply has been initialised." % Timestamp())


async def user_input():
    content = await ainput(">")
    print(content)

async def print_something():
    await asyncio.sleep(5)
    print('something')

async def loop_main():
    tasks = [user_input(), print_something()]
    await asyncio.gather(*tasks)


class AutonoMachine:
    """
    A system designed to autonomously process a machine learning task.
    """
    
    def __init__(self):
        log.info("%s - An AutonoMachine has been initialised." % Timestamp())
        self.run()
        
    async def user_input(self):
        while True:
            content = await ainput(">")
            print(content)
        
    def run(self):
        log.info("Run")
        loop = asyncio.get_event_loop()
        # asyncio.run(loop_main())
        if loop.is_running() == False:
            log.info("Out")
            asyncio.run(user_input())
        else:
            log.info("In")
            loop.create_task(user_input())
        # loop.close()