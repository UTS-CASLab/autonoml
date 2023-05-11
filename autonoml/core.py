# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:39:37 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .settings import SystemSettings as SS

import asyncio
# from aioconsole import ainput

from river import linear_model
from river import metrics


# TODO: Consider how the data is best stored, including preallocated arrays.
class DataStorage:
    """
    A collection of data that supplies machine learning processes.
    """
    
    def __init__(self):
        log.info("%s - DataStorage has been initialised." % Timestamp())
        
        self.timestamps = list()
        self.data = dict()
        
    def store_data(self, in_timestamp, in_keys, in_elements):
        
        self.timestamps.append(in_timestamp)
        
        # Extend all existing feature columns by one.
        for key in self.data:
            self.data[key].append(None)
        
        for key, element in zip(in_keys, in_elements):
            # If a new feature is identified in the keys, initialise a list.
            if not key in self.data:
                self.data[key] = [None]*len(self.timestamps)
            self.data[key][-1] = element
        

# TODO: Extend this beyond .csv files.
class DataPort:
    """
    An object to wrap up a connection to a data source.
    """
    
    def __init__(self, in_id, in_data_storage, 
                 in_host = SS.DEFAULT_HOST, in_port = SS.DEFAULT_PORT_DATA):
        log.info("%s - A DataPort has been initialised." % Timestamp())
        
        self.id = in_id
        self.data_storage = in_data_storage
        self.host = in_host
        self.port = in_port
        
        # A list of names to ID elements within incoming data.
        self.keys_data = list()
        
        self.ops = None
        self.task = asyncio.get_event_loop().create_task(self.run_connection())
        
    def close(self):
        # Cancel all asynchronous operations.
        if self.ops:
            for op in self.ops:
                op.cancel()
        
        self.task.cancel()
        
    async def run_connection(self):
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.host, self.port)
                log.warning("%s - Connected to host %s, port %s." 
                            % (Timestamp(), self.host, self.port))
                self.ops = [asyncio.create_task(op) for op in [self.send_confirm_to_server(writer),
                                                               self.receive_data_from_server(reader)]]
                for op in asyncio.as_completed(self.ops):
                    await op
                    for op_other in self.ops:
                        op_other.cancel()
                    break
                writer.close()
                await writer.wait_closed()
                    
            except Exception as e:
                log.debug(e)
                log.warning("%s - Cannot connect to host %s, port %s. Retrying." 
                            % (Timestamp(), self.host, self.port))
                
    async def send_confirm_to_server(self, in_writer):
        while True:
            in_writer.write(SS.SIGNAL_CONFIRM.encode("utf8"))
            try:
                await in_writer.drain()
            except Exception as e:
                log.warning(e)
                break
            await asyncio.sleep(SS.DELAY_FOR_CLIENT_CONFIRM)
        
    async def receive_data_from_server(self, in_reader):
        while True:
            try:
                message = await in_reader.readline()
            except Exception as e:
                log.warning(e)
                break
            data = message.decode("utf8").rstrip().split(",")
            
            # TODO: Consider checking only once if keys need to be established.
            if not self.keys_data:
                self.keys_data = [str(self.id) + "_" + str(x) for x in range(len(data))]
                
            timestamp = Timestamp()
            self.data_storage.store_data(timestamp, self.keys_data, data)
            log.info("%s - Data received: %s" % (timestamp, data))

class TaskSolver:
    """
    A wrapper for components that learn from data and respond to queries.
    """
    
    def __init__(self, in_data_storage):
        log.info("%s - A TaskSolver has been initialised." % Timestamp())
        
        self.data_storage = in_data_storage
        
        # linear_model.LogisticRegression()
        # self.metric = metrics.Accuracy()
        



class AutonoMachine:
    """
    A system designed to autonomously process a machine learning task.
    """
    
    def __init__(self):
        log.info("%s - An AutonoMachine has been initialised." % Timestamp())
        
        self.data_storage = DataStorage()
        self.data_ports = dict()
        
        self.task_solver = None
        
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
            
    async def gather_ops(self):
        self.ops = [asyncio.create_task(op) for op in [self.check_issues()]]
        await asyncio.gather(*self.ops, return_exceptions=True)
            
    def stop(self):
        log.info("%s - The AutonoMachine is now stopping." % Timestamp())
        self.is_running = False
        
        # Cancel all asynchronous operations.
        if self.ops:
            for op in self.ops:
                op.cancel()
                
        # Close all data ports.
        for id in self.data_ports:
            self.data_ports[id].close()
                
    def open_data_port(self, in_host = SS.DEFAULT_HOST, in_port = SS.DEFAULT_PORT_DATA):
        # TODO: Confirm how new data ports are named.
        id_data_port = len(self.data_ports)
        self.data_ports[id_data_port] = DataPort(in_id = id_data_port,
                                                         in_data_storage = self.data_storage,
                                                         in_host = in_host, in_port = in_port)
        
    def learn(self, in_id_target):
        self.task_solver = TaskSolver(in_data_storage = self.data_storage)
        
    # # TODO: Decide on a stop event when UI gets fleshed out.
    # async def check_stop(self):
    #     while self.is_running:
    #         await asyncio.sleep(10)
    #         # self.stop()
    
    async def check_issues(self):
        id_issue = 0
        while self.is_running:
            await asyncio.sleep(self.delay_for_issue_check)
            # TODO: Consider enums for issues.
            if id_issue in [0, 1] and not self.data_ports:
                self.warn_issue("No data ports have been assigned to the AutonoMachine.")
                id_issue = 1
            elif id_issue in [0, 2] and not self.task_solver:
                self.warn_issue("The AutonoMachine has not been given a learning task.")
                id_issue = 2
            else:
                self.delay_for_issue_check = SS.BASE_DELAY_FOR_ISSUE_CHECK
                id_issue = 0
            
            if not id_issue == 0:
                self.delay_for_issue_check *= 2
                
    def warn_issue(self, in_message):
        log.warning("%s - %i+ seconds since last check - %s" 
                    % (Timestamp(), self.delay_for_issue_check, in_message))