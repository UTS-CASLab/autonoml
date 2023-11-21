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
                 in_tags: Dict[str, str] = None):
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
        
        # # An ordered list of keys associated with elements of inflow data.
        # self.keys = None
        # self.data_types = None
        
    # TODO: Update logs for queries.
    async def ingest_file(self, in_filepath: str,
                          in_file_has_headers: bool = True, as_query: bool = False):

        log.info("%s - DataPort '%s' is ingesting a file: %s" 
                 % (Timestamp(), self.name, in_filepath))
   
        time_start = Timestamp().time

        # Read CSV into an arrow table.
        read_options = pacsv.ReadOptions(use_threads = True,
                                         autogenerate_column_names = not in_file_has_headers)
        data = pacsv.read_csv(in_filepath, read_options = read_options)

        self.data_storage.store_data(in_data = data,
                                     in_tags = self.tags,
                                     as_query = as_query)

        time_end = Timestamp().time
                
        log.info("%s - DataPort '%s' has acquired and stored %s instances of data.\n"
                 "%s   Time taken: %.3f s" 
                  % (Timestamp(), self.name, data.num_rows,
                     Timestamp(None), time_end - time_start))

                


class DataPortStream(DataPort):
    """
    A DataPort that connects to a server.
    """
    
    def __init__(self, in_data_storage, 
                 in_hostname = SS.DEFAULT_HOSTNAME, in_port = SS.DEFAULT_PORT_DATA,
                 in_id_stream: str = None, in_tags: Dict[str, str] = None):
        super().__init__(in_data_storage = in_data_storage, in_name = in_id_stream,
                         in_tags = in_tags)
        log.info("%s   This DataPort is designed for streams." % Timestamp(None))
        
        # Server details that this data port is targeting.
        self.target_hostname = in_hostname
        self.target_port = in_port
        
        # Field names for the streamed data that will be encountered.
        self.field_names = None

        # A switch for whether encountered data is currently being streamed into storage.
        self.is_storing = False
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
        
    #     self.task.cancel()

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


        
    async def run_connection(self):
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.target_hostname, self.target_port)
                log.warning("%s - DataPort '%s' is connected to host %s, port %s." 
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
                log.debug(e)
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

            data_list = message.decode("utf8").rstrip().split(",")
            if self.field_names is None:
                data_dict_list = [{self.name + "_" + str(idx): data_list[idx] for idx in range(len(data_list))}]
                data = pa.Table.from_pylist(data_dict_list)
                self.field_names = data.schema.names
            else:
                data_dict_list = [{self.field_names[idx]: data_list[idx] for idx in range(len(data_list))}]
                data = pa.Table.from_pylist(data_dict_list)
                
            timestamp = Timestamp()
            if self.is_storing:
                self.data_storage.store_data(in_data = data, 
                                            in_tags = self.tags)
            # log.debug("%s - DataPort '%s' received data: %s" % (timestamp, self.name, data))