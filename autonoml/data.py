# -*- coding: utf-8 -*-
"""
Created on Fri May 12 22:21:05 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp

import asyncio
import ast

import pandas as pd

# TODO: Redesign so the inference/conversion is done at DataPort interface.
#       This will allow CSV text file inputs to be treated differently from other inputs.
def infer_data_type(in_element):
    data_type = type(in_element)
    
    # Check if a string data type can be converted to something else.
    if data_type == str:
        try:
            data_type = type(ast.literal_eval(in_element))
        except:
            pass
    
    return data_type

# TODO: Consider how the data is best stored, including preallocated arrays.
class DataStorage:
    """
    A collection of data that supplies machine learning processes.
    """
    
    def __init__(self):
        log.info("%s - DataStorage has been initialised." % Timestamp())
        
        self.timestamps = list()
        self.data = dict()          # Stored data arranged in keyed lists.
        self.data_types = dict()    # The data types for each keyed list.
        # Note: Data types are actual types not strings.
        
        # Ingested data arrives from data ports.
        # For data port X, this data is sent as a list of elements.
        # The elements have keys: X_0, X_1, etc.
        # Define a dict that links port-specific keys to storage-specific keys.
        # This directs elements of incoming data to the right list.
        self.keys_port_to_storage = dict()
        
        # Set up a variable that can be awaited elsewhere.
        # This 'switch', when flicked, signals learners to ingest new data.
        self.has_new_data = asyncio.Future()
        
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
                log.info("%s - DataStorage is newly encountering data "
                         "from a DataPort with key '%s'." 
                         % (Timestamp(), key_port))
                self.keys_port_to_storage[key_port] = key_port
            
            key_storage = self.keys_port_to_storage[key_port]
            
            if not key_storage in self.data:
                log.info("%s - DataStorage has begun storing data "
                         "in a list with key '%s'." 
                         % (Timestamp(), key_storage))
                self.data[key_storage] = [None]*len(self.timestamps)
                
                # The first element in a list determines its data type.
                self.data_types[key_storage] = infer_data_type(element)
            
            # Add the new element to the list with str-to-type conversion.
            # Note the function call.
            try:
                # TODO: Some data types do not convert, e.g. NoneType. Consider how to fix/avoid.
                self.data[key_storage][-1] = self.data_types[key_storage](element)
            except Exception as e:
                # TODO: Handle changes in data type for messy datasets.
                raise e
        
        # Flick a switch so that learners can start ingesting new data.
        # Note: Resolving awaited futures are priority microtasks.
        # The following reset runs after the learners are signalled.
        self.has_new_data.set_result(True)
        self.has_new_data = asyncio.Future()
        
    def info(self):
        """
        Utility method to give user info about data ports and storage.
        """
        log.info("Stored data is arranged into lists identified as follows.")
        log.info("Keys: %s" % ", ".join(key + " (" + self.data_types[key].__name__ + ")" 
                                        for key in self.data_types))
        log.info("DataPorts pipe data into the lists as follows.")
        log.info("Pipe: %s" % ", ".join("{" + key + " -> " + self.keys_port_to_storage[key] + "}" 
                                        for key in self.keys_port_to_storage))
        
    def update(self, in_keys_port, in_keys_storage):
        """
        Take in a list of DataPort data keys.
        Update the DataStorage lists into which the data is directed.
        This may be as simple as renaming lists or may involve merging.
        Note: This process also shifts historically stored data across.
        """
        
        for count_key in range(len(in_keys_port)):
            key_port = in_keys_port[count_key]
            key_storage = in_keys_storage[count_key]
            
            if not key_port in self.keys_port_to_storage:
                log.warning("%s - No existing DataPort keys data with '%s'. "
                            "Ignoring update request: {%s -> %s}"
                            % (Timestamp(), key_port, key_port, key_storage))
            else:
                key_storage_old = self.keys_port_to_storage[key_port]
                data_type_old = self.data_types[key_storage_old]
                
                # Handle merging with pre-existing lists.
                if key_storage in self.data:
                    data_type_existing = self.data_types[key_storage]
                    
                    # Refuse for clashing data types.
                    # TODO: Consider allowing it but generalising data types.
                    if not data_type_existing == data_type_old:
                        log.warning("%s - DataStorage already contains list '%s' with type '%s'. "
                                    "Refusing to merge in list '%s' with type '%s'."
                                    % (Timestamp(), key_storage, data_type_existing.__name__, 
                                       key_storage_old, data_type_old.__name__))
                    else:
                        log.warning("%s - DataStorage already contains list '%s'. "
                                    "Proceeding to overwrite with list '%s' where values exist."
                                    % (Timestamp(), key_storage, key_storage_old))
                        self.data_types.pop(key_storage_old)
                        list_old = self.data.pop(key_storage_old)
                        for count_element in range(len(list_old)):
                            element = list_old[count_element]
                            if element is not None:
                                self.data[key_storage][count_element] = element
                        
                        # Update directions for the DataPort.
                        self.keys_port_to_storage[key_port] = key_storage
                        
                # Handle what is effectively the renaming of a list.
                else:
                    self.data_types[key_storage] = self.data_types.pop(key_storage_old)
                    self.data[key_storage] = self.data.pop(key_storage_old)
                    
                    # Update directions for the DataPort.
                    self.keys_port_to_storage[key_port] = key_storage
                
        
        
    def get_dataframe(self):
        """
        A utility method converting data dictionary into a Pandas dataframe.
        This is slow and should be called sparingly.
        """
        df = pd.DataFrame.from_dict(self.data)
        df.index = self.timestamps
        return df