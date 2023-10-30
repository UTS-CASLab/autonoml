# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:00:56 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .settings import SystemSettings as SS

from .data_storage import DataStorage

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

    def __init__(self, in_data_storage: DataStorage):
        self.name = "Port_" + str(DataPort.count)
        DataPort.count += 1
        log.info("%s - Initialising DataPort '%s'." % (Timestamp(), self.name))
        
        # Reference to the DataStorage contained in the AutonoMachine.
        self.data_storage = in_data_storage
        
        # An ordered list of keys associated with elements of inflow data.
        self.keys = None
        self.data_types = None
        
    # TODO: Update logs for queries.
    async def ingest_file(self, in_filepath: str, in_tags = None,
                          in_file_has_headers: bool = True, as_query: bool = False):

        log.info("%s - DataPort '%s' is ingesting a file: %s" 
                 % (Timestamp(), self.name, in_filepath))
   
        time_start = Timestamp().time

        # Read CSV into an arrow table.
        read_options = pacsv.ReadOptions(use_threads = True,
                                         autogenerate_column_names = not in_file_has_headers)
        data = pacsv.read_csv(in_filepath, read_options = read_options)

        # Ensure tags are in string format.
        tags = dict()
        if not in_tags is None:
            for key in in_tags:
                tags[str(key)] = str(in_tags[key])

        self.data_storage.store_data(in_data = data,
                                     in_tags = tags,
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
    
    def __init__(self, in_id, in_data_storage, 
                 in_hostname = SS.DEFAULT_HOSTNAME, in_port = SS.DEFAULT_PORT_DATA):
        super().__init__(in_id = in_id, in_data_storage = in_data_storage)
        log.info("%s   This DataPort is designed for streams." % Timestamp(None))
        
        # Server details that this data port is targeting.
        self.target_hostname = in_hostname
        self.target_port = in_port
        
        self.ops = None
        self.task = asyncio.get_event_loop().create_task(self.run_connection())
        
    def __del__(self):
        # Cancel all asynchronous operations.
        if self.ops:
            for op in self.ops:
                op.cancel()
        
        self.task.cancel()
        
    async def run_connection(self):
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.target_hostname, self.target_port)
                log.warning("%s - DataPort '%s' is connected to host %s, port %s." 
                            % (Timestamp(), self.name, self.target_hostname, self.target_port))
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
            data = message.decode("utf8").rstrip().split(",")
                
            timestamp = Timestamp()
            self.data_storage.store_data(in_timestamp = timestamp, 
                                         in_elements = data, 
                                         in_port_id = self.name)
            log.info("%s - DataPort '%s' received data: %s" % (timestamp, self.name, data))