# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 21:38:26 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .settings import SystemSettings as SS

import asyncio

# import multiprocessing as mp

# import time

# def stream_data():
#     while True:
#         time.sleep(2)
#         log.info("WHOOP WHOOP!")

# class SimDataStreamer:
#     """
#     An object that simulates a data-streaming server in a new process.
#     """
    
#     def __init__(self):
#         log.info("%s - A SimDataStreamer has been initialised." % Timestamp())
        
#         self.process = mp.Process(target=stream_data)
#         self.process.start()
        
#     def stop(self):
#         log.info("%s - The SimDataStreamer is now stopping." % Timestamp())
        
#         self.process.join(0)
#         self.process.terminate()
        
#         # self.data_storage = DataStorage()
#         # self.data_ports = dict()
        
#         # self.is_running = False
#         # self.delay_until_check = SS.BASE_DELAY_UNTIL_CHECK
        
#         # self.ops = None
        
#         # self.run()
        
# #     def run(self):
# #         log.info("%s - The AutonoMachine is now running." % Timestamp())
# #         self.is_running = True
        
# #         # Check the Python environment for an asynchronous event loop.
# #         # Gather operations and hand them to a new/existing event loop.
# #         loop = asyncio.get_event_loop()
# #         if loop.is_running() == False:
# #             log.debug(("No asyncio event loop is currently running.\n"
# #                        "One will be launched for AutonoML operations."))
# #             asyncio.run(self.gather_ops())
# #         else:
# #             log.debug(("The Python environment is already running an asyncio event loop.\n"
# #                        "It will be used for AutonoML operations."))
# #             loop.create_task(self.gather_ops())


# # # a custom function that blocks for a moment
# # def task():
# #     # block for a moment
# #     sleep(1)
# #     # display a message
# #     print('This is from another process')

class SimDataStreamer:
    """
    An object that simulates a data-streaming server.
    """
    
    def __init__(self, 
                 in_filename_data = None,
                 in_period_data_stream = SS.PERIOD_DATA_STREAM,
                 in_file_has_headers = True):
        log.info("%s - A SimDataStreamer has been initialised." % Timestamp())
        
        # self.data_storage = DataStorage()
        # self.data_ports = dict()
        
        # self.delay_until_check = SS.BASE_DELAY_UNTIL_CHECK
        
        self.filename_data = in_filename_data
        self.file_has_headers = in_file_has_headers
        self.data = asyncio.Future()
        
        self.server = None
        
        self.ops = None
        
        self.is_running = False
        self.run()
        
    def run(self):
        log.info("%s - The SimDataStreamer is now running." % Timestamp())
        self.is_running = True
        
        # Check the Python environment for an asynchronous event loop.
        # Gather operations and hand them to a new/existing event loop.
        loop = asyncio.get_event_loop()
        if loop.is_running() == False:
            log.debug(("No asyncio event loop is currently running.\n"
                       "One will be launched for simulated data streaming."))
            asyncio.run(self.gather_ops())
        else:
            log.debug(("The Python environment is already running an asyncio event loop.\n"
                       "It will be used for simulated data streaming."))
            loop.create_task(self.gather_ops())
            
    def stop(self):
        log.info("%s - The SimDataStreamer is now stopping." % Timestamp())
        self.is_running = False
        
        # Cancel all asynchronous operations.
        if self.ops:
            for op in self.ops:
                op.cancel()
        
        # Close server.
        if self.server:
            self.server.close()
            
    async def gather_ops(self):
        self.ops = [asyncio.create_task(op) for op in [self.run_server(),
                                                       self.get_data(),
                                                       self.check_stop()]]
        await asyncio.gather(*self.ops, return_exceptions=True)
    
    async def handle_client(self, reader, writer):
        # time_last_confirm = Timestamp().time
        # while Timestamp().time < time_last_confirm + SS.DELAY_FOR_SOCKET_TIMEOUT:
        #     await asyncio.sleep(SS.PERIOD_DATA_STREAM)
        #     response = "WHOOP\n"
        #     print(response)
        #     writer.write(response.encode("utf8"))
        #     await writer.drain()
        while True:
            data = await self.data
            writer.write(data.encode("utf8"))
            try:
                await writer.drain()
            except Exception as e:
                log.warning(e)
                log.warning("%s - SimDataStreamer has lost connection with a client." % Timestamp())
                break
        writer.close()
        await writer.wait_closed()

    async def run_server(self):
        self.server = await asyncio.start_server(self.handle_client, SS.DEFAULT_HOST, SS.DEFAULT_PORT)
        async with self.server:
            await self.server.serve_forever()
            
    async def get_data(self):
        try:
            with open(self.filename_data, "r") as data_file:
                if self.file_has_headers:
                    data_file.readline()
                while True:
                    line = data_file.readline()
                    log.info("%s - Data acquired: %s" % (Timestamp(), line))
                    if not line:
                        break
                    
                    # Resolving awaited future results are priority microtasks.
                    self.data.set_result(line)
                    self.data = asyncio.Future()
                    await asyncio.sleep(SS.PERIOD_DATA_STREAM)
        except:
            pass
        if self.is_running:
            log.warning("%s - SimDataStreamer cannot find data to read from a valid file." % Timestamp())
        
    async def check_stop(self):
        while self.is_running:
            await asyncio.sleep(60)
            self.stop()
        
    # async def check_issues(self):
    #     while self.is_running:
    #         await asyncio.sleep(self.delay_until_check)
    #         is_issue = False
    #         if not self.data_ports:
    #             log.warning(("%s - %i+ seconds since last check - "
    #                          "No data ports have been assigned to the AutonoMachine.") 
    #                         % (Timestamp(), self.delay_until_check))
    #             is_issue = True
                
    #         if is_issue:
    #             self.delay_until_check *= 2
    #         else:
    #             self.delay_until_check = SS.BASE_DELAY_UNTIL_CHECK