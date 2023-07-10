# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:00:56 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .settings import SystemSettings as SS

import asyncio

# TODO: Extend this beyond .csv files.
class DataPort:
    """
    An object to wrap up a connection to a data source.
    """
    
    def __init__(self, in_id, in_data_storage):
        log.info("%s - DataPort '%s' has been initialised." % (Timestamp(), in_id))
        
        self.id = in_id     # String to id data port.
        
        # Reference to the DataStorage contained in the AutonoMachine.
        self.data_storage = in_data_storage
        
        # An ordered list of keys associated with elements of inflow data.
        self.keys = None
        
    # TODO: Consider Pandas if it is faster.
    def ingest_file(self, in_filepath, in_file_has_headers = True):
        
        log.info("%s - DataPort '%s' is ingesting a file: %s" 
                 % (Timestamp(), self.id, in_filepath))
        
        # with open(in_filepath, "r") as data_file:
        #     if not in_file_has_headers:
        #         with open(in_filepath, "r") as temp_file:
        #             num_fields = len(temp_file.readline().rstrip().split(","))
        #         self.keys = [str(count_field) for count_field in range(num_fields)]
        #     it = zip(*csv.DictReader(data_file, fieldnames = self.keys))
        #     data = {el[0]: [val for val in el[1:]] for el in it}
        #     self.data_storage.store_data(in_timestamp = Timestamp(),
        #                                  in_data_port_id = self.id,
        #                                  in_data_dict = data)
        #     num_instances = len(data[list(test_dict.keys())[0]])
        #     log.info("%s - DataPort '%s' has acquired %i instances of data." 
        #              % (Timestamp(), self.id, num_instances))
        
        with open(in_filepath, "r") as data_file:
            # If there are headers, these become the inflow keys.
            if in_file_has_headers:
                line = data_file.readline()
                self.keys = line.rstrip().split(",")

            count_instance = 0
            for line in data_file:
                data = line.rstrip().split(",")
                
                # If there are no headers...
                # Inflow keys enumerate the inflow data elements encountered.
                if count_instance == 0 and not in_file_has_headers:
                    self.keys = [str(num_element) for num_element in range(len(data))]
                    
                self.data_storage.store_data(in_timestamp = Timestamp(),
                                              in_data_port_id = self.id,
                                              in_keys = self.keys,
                                              in_elements = data)
                count_instance += 1
                
        log.info("%s - DataPort '%s' has acquired and stored %s instances of data." 
                  % (Timestamp(), self.id, count_instance))
                


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
                            % (Timestamp(), self.id, self.target_hostname, self.target_port))
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
                            % (Timestamp(), self.id, self.target_hostname, self.target_port))
                
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
                                         in_port_id = self.id)
            log.info("%s - DataPort '%s' received data: %s" % (timestamp, self.id, data))