# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 21:38:26 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .settings import SystemSettings as SS

from copy import deepcopy

import asyncio

class SimDataStreamer:
    """
    An object that simulates data-streaming servers based on a CSV file.
    The primary server generates training data.
    The secondary server generates queries with expected responses.
    In practice the data instances are simply lines from the CSV file.
    
    Per broadcasting cycle, the primary transmits 'instances_per_query' lines.
    The secondary then transmits one line while the primary takes a break.
    Non-integer ratios are allowed, which the broadcast averages to over time.
    The lines are broadcast with a period of 'period_data_stream'.
    """
    
    def __init__(self, 
                 in_filename_data = None,
                 in_instances_per_query = 1,
                 in_period_data_stream = SS.PERIOD_DATA_STREAM,
                 in_file_has_headers = True):
        log.info("%s - A SimDataStreamer has been initialised." % Timestamp())
        
        self.filename_data = in_filename_data
        self.file_has_headers = in_file_has_headers
        self.instances_per_query = in_instances_per_query
        self.period_data_stream = in_period_data_stream
        
        self.ops = None
        self.server_data = None
        self.server_query = None
        
        # 'Future' objects that will be updated with data/queries to broadcast.
        self.data = asyncio.Future()
        self.query = asyncio.Future()
        
        # A timestamp used to track the last confirmation of client connection.
        # This is a 'global' value, in case multiple clients connect.
        self.timestamp_confirm_global = Timestamp()
        
        self.run()
        
    def run(self):
        log.info("%s - The SimDataStreamer is now running." % Timestamp())
        
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
            
    async def gather_ops(self):
        """
        Gather top-level concurrent tasks.
        These include...
        - Running the data broadcasting server.
        - Generating the data to broadcast.
        - Checking when to shut down the server.
        """
        
        self.ops = [asyncio.create_task(op) for op in [self.run_server(),
                                                       self.get_data(),
                                                       self.check_stop()]]
        await asyncio.gather(*self.ops, return_exceptions=True)
        
    async def check_stop(self):
        """
        The streamer stops if there is no interest in its server for a while.
        """
        while Timestamp().time - self.timestamp_confirm_global.time < SS.DELAY_FOR_SHUTDOWN_CONFIRM:
            await asyncio.sleep(SS.PERIOD_SHUTDOWN_CHECK)
        log.warning("%s - No clients have confirmed connection to the server "
                    "in over %s seconds." % (Timestamp(), SS.DELAY_FOR_SHUTDOWN_CONFIRM))
        self.stop()
        
    def stop(self):
        log.info("%s - The SimDataStreamer is now stopping." % Timestamp())
        
        # Cancel all asynchronous operations.
        if self.ops:
            for op in self.ops:
                op.cancel()
        
        # Close servers.
        if self.server_data:
            self.server_data.close()
        if self.server_query:
            self.server_query.close()

    #%% Server-client process management.

    async def run_server(self):
        self.server_data = await asyncio.start_server(self.handle_client, 
                                                      SS.DEFAULT_HOST, SS.DEFAULT_PORT_DATA)
        self.server_query = await asyncio.start_server(lambda r, w: self.handle_client(r, w, is_query = True), 
                                                       SS.DEFAULT_HOST, SS.DEFAULT_PORT_QUERY)
        async with self.server_data, self.server_query:
            await asyncio.gather(self.server_data.serve_forever(), self.server_query.serve_forever())

    async def handle_client(self, in_reader, in_writer, is_query = False):
        """
        If a client connects, regularly check for connection confirmations.
        While connected, transmit data.
        """
        log.info("%s - SimDataStreamer has established connection with a client." % Timestamp())
        timestamp_confirm_local = Timestamp()
        ops = [asyncio.create_task(op) for op in [self.send_data_to_client(in_writer, timestamp_confirm_local, is_query),
                                                  self.receive_confirm_from_client(in_reader, timestamp_confirm_local)]]
        for op in asyncio.as_completed(ops):
            await op
            for op_other in ops:
                op_other.cancel()
            break
        log.warning("%s - SimDataStreamer has lost connection with a client." % Timestamp())
        in_writer.close()
        await in_writer.wait_closed()
        
    async def send_data_to_client(self, in_writer, in_timestamp_confirm, is_query):
        while Timestamp().time - in_timestamp_confirm.time < SS.DELAY_FOR_SERVER_ABANDON:
            # If the 'future' object is updated with data, proceed to transmit.
            if is_query:
                message = await self.query
            else:
                message = await self.data
            in_writer.write(message.encode("utf8"))
            try:
                await in_writer.drain()
            except Exception as e:
                log.warning(e)
                break
            
    async def receive_confirm_from_client(self, in_reader, in_timestamp_confirm):
        while Timestamp().time - in_timestamp_confirm.time < SS.DELAY_FOR_SERVER_ABANDON:
            try:
                # Any message from the client with an endline confirms the connection.
                await in_reader.readline()
            except Exception as e:
                log.warning(e)
                break
            in_timestamp_confirm.update_from(Timestamp())
            self.timestamp_confirm_global = deepcopy(in_timestamp_confirm)

    #%% Data generation for transmission to clients.
            
    async def get_data(self):
        """
        Iteratively reads the lines of a text file.
        Each line is sent to any connected clients elsewhere.
        Training/testing ratio is set by 'instances_per_query'.
        The file is assumed to be in CSV format. 
        """
        
        try:
            with open(self.filename_data, "r") as data_file:
                # Ignore the header line.
                if self.file_has_headers:
                    data_file.readline()
                count_instance = 0
                while True:
                    line = data_file.readline()
                    if not line:
                        break
                    
                    # Stream writers for connected clients wait for a 'future'.
                    # Set the result so they can transmit the data.
                    # Note: Resolving awaited futures are priority microtasks.
                    # The following reset runs after the writers get results.
                    if count_instance < self.instances_per_query:
                        log.info("%s - Data generated: %s" % (Timestamp(), line.rstrip()))
                        self.data.set_result(line)
                        self.data = asyncio.Future()
                        count_instance += 1
                    else:
                        log.info("%s - Query (and expected response) generated: %s" % (Timestamp(), line.rstrip()))
                        self.query.set_result(line)
                        self.query = asyncio.Future()
                        # Allow non-integer train/test ratios by subtracting.
                        count_instance -= self.instances_per_query
                    await asyncio.sleep(self.period_data_stream)
        except asyncio.CancelledError:
            pass
        except:
            log.warning("%s - SimDataStreamer cannot find data to read from a valid file." % Timestamp())