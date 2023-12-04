# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:00:56 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp, identify_exception
from .settings import SystemSettings as SS
from .concurrency import create_async_task_from_sync, create_async_task, schedule_this

from .data_storage import DataStorage

from typing import Dict, List

import asyncio

import pyarrow as pa
from pyarrow import csv as pacsv

def read_csv_to_arrow(in_filename):
    # Specify the CSV file reader options
    read_options = pa.csv.ReadOptions(column_types=None, use_threads=True)
    
    # Read the CSV file into an Arrow Table
    table = pa.csv.read_csv(in_filename, read_options=read_options)
    return table

# TODO: Extend this beyond .csv files.
class DataPort:
    """
    An object to wrap up a connection to a data source.
    """
    
    count = 0

    def __init__(self, in_data_storage: DataStorage, in_name: str = None, 
                 in_tags: Dict[str, str] = None, is_for_queries: bool = False):
        self.name = "Port_" + str(DataPort.count)
        DataPort.count += 1
        if not in_name is None:
            self.name = in_name
        log.info("%s - Initialising DataPort '%s'." % (Timestamp(), self.name))
        
        # Reference to the DataStorage contained in the AutonoMachine.
        self.data_storage = in_data_storage

        # Ensure tags are in string format.
        self.tags = dict()
        if not in_tags is None:
            for key in in_tags:
                self.tags[str(key)] = str(in_tags[key])
        
        self.is_for_queries = is_for_queries
        
    # TODO: Update logs for queries.
    async def ingest_file(self, in_filepath: str, in_file_has_headers: bool = True):

        log.info("%s - DataPort '%s' is ingesting a file: %s" 
                 % (Timestamp(), self.name, in_filepath))
   
        time_start = Timestamp().time

        # Read CSV into an arrow table.
        read_options = pacsv.ReadOptions(use_threads = True,
                                         autogenerate_column_names = not in_file_has_headers)
        data = pacsv.read_csv(in_filepath, read_options = read_options)

        self.data_storage.store_data(in_data = data,
                                     in_tags = self.tags,
                                     as_query = self.is_for_queries)

        time_end = Timestamp().time
                
        log.info("%s - DataPort '%s' has acquired and stored %s instances of data.\n"
                 "%s   Time taken: %.3f s" 
                  % (Timestamp(), self.name, data.num_rows,
                     Timestamp(None), time_end - time_start))
        
    def __del__(self):
        log.debug("Finalising DataPort '%s'." % self.name)

                


class DataPortStream(DataPort):
    """
    A DataPort that connects to a server.
    """
    
    def __init__(self, in_data_storage, 
                 in_hostname = SS.DEFAULT_HOSTNAME, in_port = SS.DEFAULT_PORT_OBSERVATIONS,
                 in_field_names: List[str] = None,
                 in_id_stream: str = None, in_tags: Dict[str, str] = None,
                 is_for_queries: bool = False):
        super().__init__(in_data_storage = in_data_storage, in_name = in_id_stream,
                         in_tags = in_tags, is_for_queries = is_for_queries)
        log.info("%s   This DataPort is designed for streams." % Timestamp(None))
        
        # Server details that this data port is targeting.
        self.target_hostname = in_hostname
        self.target_port = in_port

        self.connection_state = False
        
        # Field names for the streamed data that will be encountered.
        self.field_names = None
        if not in_field_names is None:
            self.field_names = in_field_names

        # A switch for whether encountered data is currently being streamed into storage.
        # It will be flicked on only if field names are provided by the user.
        # Otherwise, the user may want to inspect the streamed data first, manually deciding on field names.
        self.is_storing = False
        if not self.field_names is None:
            self.is_storing = True
        if not self.is_storing:
            log.info("%s   It is currently not storing any streamed data it may encounter.\n"
                     "%s   Do not toggle storage until field names are satisfactorily set." 
                     % (Timestamp(None), Timestamp(None)))

        self.ops = None

    def __del__(self):
        log.debug("Finalising stream-based DataPort '%s'." % self.name)

    # def __del__(self):
    #     # Cancel all asynchronous operations.
    #     if self.ops:
    #         for op in self.ops:
    #             op.cancel()



    def get_field_names(self):
        return self.field_names

    @schedule_this
    async def set_field_names(self, in_field_names: List[str]):
        self.field_names = in_field_names

    @schedule_this
    async def toggle_storage(self):
        self.is_storing = not self.is_storing
        if self.is_storing:
            log.info("%s - DataPort '%s' has begun storing streamed data." % (Timestamp(), self.name))
        else:
            log.info("%s - DataPort '%s' has stopped storing streamed data." % (Timestamp(), self.name))

    def is_connected(self):
        return self.connection_state
    

        
    async def run_connection(self):
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.target_hostname, self.target_port)
                self.connection_state = True    # TODO: Work out where to robustly mark this as false.
                log.info("%s - DataPort '%s' is connected to host %s, port %s."
                         % (Timestamp(), self.name, self.target_hostname, self.target_port))
                self.ops = [create_async_task(self.send_confirm_to_server, writer),
                            create_async_task(self.receive_data_from_server, reader)]
                for op in asyncio.as_completed(self.ops):
                    await op
                    for op_other in self.ops:
                        op_other.cancel()
                    break
                writer.close()
                await writer.wait_closed()
                    
            except Exception as e:
                identify_exception(e, "")
                if not self.ops is None:
                    for op in self.ops:
                        op.cancel()
                log.warning("%s - DataPort '%s' cannot connect to host %s, port %s. Retrying." 
                            % (Timestamp(), self.name, self.target_hostname, self.target_port))
                
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

            data_buffer = pa.py_buffer(message)
            input_stream = pa.input_stream(data_buffer)
            # TODO: Check if options should be defined once for performance.
            if self.field_names is None:
                read_options = pacsv.ReadOptions(use_threads = True, autogenerate_column_names = True)
                data = pacsv.read_csv(input_stream, read_options = read_options)
                self.field_names = data.schema.names
            else:
                read_options = pacsv.ReadOptions(use_threads = True, column_names = self.field_names)
                # convert_options = pacsv.ConvertOptions(include_columns = self.field_names, include_missing_columns = True)
                data = pacsv.read_csv(input_stream, read_options = read_options) #, convert_options = convert_options)
                
            if self.is_storing:
                self.data_storage.store_data(in_data = data, 
                                            in_tags = self.tags,
                                            as_query = self.is_for_queries)
            # log.debug("%s - DataPort '%s' received data: %s" % (Timestamp(), self.name, data))