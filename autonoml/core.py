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



class DataStorage:
    """
    A collection of data that supplies machine learning processes.
    """
    
    def __init__(self):
        log.info("%s - DataStorage has been initialised." % Timestamp())


# https://stackoverflow.com/questions/48506460/python-simple-socket-client-server-using-asyncio
# import asyncio, socket

# async def handle_client(reader, writer):
#     request = None
#     while request != 'quit':
#         request = (await reader.read(255)).decode('utf8')
#         response = str(eval(request)) + '\n'
#         writer.write(response.encode('utf8'))
#         await writer.drain()
#     writer.close()

# async def run_server():
#     server = await asyncio.start_server(handle_client, 'localhost', 15555)
#     async with server:
#         await server.serve_forever()

# asyncio.run(run_server())



class AutonoMachine:
    """
    A system designed to autonomously process a machine learning task.
    """
    
    def __init__(self):
        log.info("%s - An AutonoMachine has been initialised." % Timestamp())
        
        self.data_storage = DataStorage()
        self.data_ports = dict()
        
        self.is_running = False
        self.delay_until_check = SS.BASE_DELAY_UNTIL_CHECK
        
        self.ops = None
        
        self.run()
        
    def run(self):
        log.info("%s - The AutonoMachine is now running." % Timestamp())
        self.is_running = True
        
        # Check the Python environment for an asynchronous event loop.
        # Gather operations and hand them to a new/existing event loop.
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
        
    # TODO: Decide on a stop event when UI gets fleshed out.
    async def check_stop(self):
        while self.is_running:
            await asyncio.sleep(10)
            # self.stop()
        
    async def check_issues(self):
        while self.is_running:
            await asyncio.sleep(self.delay_until_check)
            is_issue = False
            if not self.data_ports:
                log.warning(("%s - %i+ seconds since last check - "
                             "No data ports have been assigned to the AutonoMachine.") 
                            % (Timestamp(), self.delay_until_check))
                is_issue = True
                
            if is_issue:
                self.delay_until_check *= 2
            else:
                self.delay_until_check = SS.BASE_DELAY_UNTIL_CHECK