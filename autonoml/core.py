# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:39:37 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .settings import SystemSettings as SS

import asyncio
# from aioconsole import ainput



class DataStorage:
    """
    A collection of data that supplies machine learning processes.
    """
    
    def __init__(self):
        log.info("%s - DataStorage has been initialised." % Timestamp())
        
class DataPort:
    """
    An object to wrap up a connection to a data source.
    """
    
    def __init__(self, in_data_storage, in_host = SS.DEFAULT_HOST, in_port = SS.DEFAULT_PORT):
        log.info("%s - A DataPort has been initialised." % Timestamp())
        
        self.data_storage = in_data_storage
        self.host = in_host
        self.port = in_port
        
        self.reader = None
        self.writer = None
        
        self.is_running = False
        self.task = asyncio.get_event_loop().create_task(self.run_connection())
        
    def close(self):
        self.is_running = False
        self.task.cancel()
        
    async def run_connection(self):
        self.is_running = True
        while self.is_running:
            try:
                self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
                
                while True:
                    message = await self.reader.readline()
                    data = message.decode("utf8")
                    log.info("%s - Data received: %s" % (Timestamp(), data))
                    
            except Exception as e:
                log.debug(e)
                log.warning("%s - Cannot connect to host %s, port %s. Retrying." 
                            % (Timestamp(), self.host, self.port))
                
        
        # self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        # print("connected")
        
        # while True:
        #     message = await self.reader.readline()
        #     data = message.decode("utf8")
        #     log.info("%s - Data received: %s" % (Timestamp(), data))
        
        # self.writer.close()
        # await self.writer.wait_closed()
        
    #     asyncio.get_event_loop().create_task(self.get_data(reader))
        
    #     self.is_running = False
    #     self.run()
        
    # def run(self):
    #     log.info("%s - The AutonoMachine is now running." % Timestamp())
    #     self.is_running = True
        
    #     # Check the Python environment for an asynchronous event loop.
    #     # Gather operations and hand them to a new/existing event loop.
    #     loop = asyncio.get_event_loop()
    #     if loop.is_running() == False:
    #         log.debug(("No asyncio event loop is currently running.\n"
    #                    "One will be launched for AutonoML operations."))
    #         asyncio.run(self.gather_ops())
    #     else:
    #         log.debug(("The Python environment is already running an asyncio event loop.\n"
    #                    "It will be used for AutonoML operations."))
    #         loop.create_task(self.gather_ops())


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
        self.data_ports = list()
        
        self.delay_for_issue_check = SS.BASE_DELAY_FOR_ISSUE_CHECK
        
        self.ops = None
        
        self.is_running = False
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
                
        # Close all data ports.
        for data_port in self.data_ports:
            data_port.close()
                
    def open_data_port(self, in_host = SS.DEFAULT_HOST, in_port = SS.DEFAULT_PORT):
        self.data_ports.append(DataPort(in_data_storage = self.data_storage, 
                                        in_host = in_host, in_port = in_port))
        
    # async def get_data(self, reader):
    #     # print('Send: %r' % message)
    #     # writer.write(message.encode())

    #     data = await reader.readline()
    #     print('Received: %s' % data.decode())
            
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
            await asyncio.sleep(self.delay_for_issue_check)
            is_issue = False
            if not self.data_ports:
                log.warning(("%s - %i+ seconds since last check - "
                             "No data ports have been assigned to the AutonoMachine.") 
                            % (Timestamp(), self.delay_for_issue_check))
                is_issue = True
                
            if is_issue:
                self.delay_for_issue_check *= 2
            else:
                self.delay_for_issue_check = SS.BASE_DELAY_FOR_ISSUE_CHECK