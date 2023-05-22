# -*- coding: utf-8 -*-
"""
Created on Fri May 12 22:21:05 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp

import pandas as pd

# TODO: Consider how the data is best stored, including preallocated arrays.
class DataStorage:
    """
    A collection of data that supplies machine learning processes.
    """
    
    def __init__(self):
        log.info("%s - DataStorage has been initialised." % Timestamp())
        
        self.timestamps = list()
        self.data = dict()
        
        # Ingested data arrives from data ports.
        # For data port X, this data is sent as a list of elements.
        # The elements have keys: X_0, X_1, etc.
        # Define a dict that links port-specific keys to storage-specific keys.
        # This determines where elements of incoming data are stored.
        self.keys_port_to_storage = dict()
        
    def store_data(self, in_timestamp, in_elements, in_port_id):
        
        self.timestamps.append(in_timestamp)
        
        # Extend all existing data lists by one empty slot.
        for key_storage in self.data:
            self.data[key_storage].append(None)
        
        for num_element, element in zip(range(len(in_elements)), in_elements):
            
            key_port = in_port_id + "_" + str(num_element)
            
            # If a new port-specific key is encountered, initialise a list.
            # The list is initially named identically to this key.
            if not key_port in self.keys_port_to_storage:
                log.info("%s - Data is being newly stored in a list with key '%s'." 
                         % (Timestamp(), key_port))
                self.keys_port_to_storage[key_port] = key_port
            
            key_storage = self.keys_port_to_storage[key_port]
            
            if not key_storage in self.data:
                self.data[key_storage] = [None]*len(self.timestamps)
            self.data[key_storage][-1] = element
        
    def info(self):
        """
        Utility method to give user info about data ports and storage.
        """
        log.info("Stored data is arranged into lists identified as follows.")
        log.info("Keys: %s" % ", ".join(self.data.keys()))
        log.info("DataPorts pipe data into the lists as follows.")
        log.info("Pipe: %s" % ", ".join("{" + key + " -> " + self.keys_port_to_storage[key]+ "}" 
                                        for key in self.keys_port_to_storage))
        
    def update(self, in_keys_port, in_keys_storage):
        
        for count_key in range(len(in_keys_port)):
            key_port = in_keys_port[count_key]
            key_storage = in_keys_storage[count_key]
            
            if not key_port in self.keys_port_to_storage:
                log.warning("%s - No existing DataPort keys data with '%s'. "
                            "Ignoring update request: {%s -> %s}"
                            % (Timestamp(), key_port, key_port, key_storage))
            else:
                key_storage_old = self.keys_port_to_storage[key_port]
                self.keys_port_to_storage[key_port] = key_storage
                if key_storage in self.data:
                    log.warning("%s - DataStorage already contains list '%s'. "
                                "Proceeding to overwrite with list '%s' where values exist."
                                % (Timestamp(), key_storage_old, key_storage))
                    list_old = self.data.pop(key_storage_old)
                    for count_element in range(len(list_old)):
                        element = list_old[count_element]
                        if element:
                            self.data[key_storage][count_element] = element
                else:
                    self.data[key_storage] = self.data.pop(key_storage_old)
                
        
        
    def get_dataframe(self):
        """
        A utility method converting data dictionary into a Pandas dataframe.
        This is slow and should be called sparingly.
        """
        df = pd.DataFrame.from_dict(self.data)
        df.index = self.timestamps
        return df