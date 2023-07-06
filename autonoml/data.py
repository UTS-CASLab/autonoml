# -*- coding: utf-8 -*-
"""
Created on Fri May 12 22:21:05 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .settings import SystemSettings as SS

import asyncio
import ast

import pandas as pd
import numpy as np

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
        self.ikeys_to_dkeys = dict()
        
        # Set up a variable that can be awaited elsewhere.
        # This 'switch', when flicked, signals learners to ingest new data.
        self.has_new_data = asyncio.Future()
        
    def store_data(self, in_timestamp, in_data_port_id, in_keys, in_elements):
        
        self.timestamps.append(in_timestamp)
        
        # Extend all existing data lists by one empty slot.
        for dkey in self.data:
            self.data[dkey].append(None)
        
        count_ikey_new = 0
        count_dkey_new = 0
        for key, element in zip(in_keys, in_elements):
            
            ikey = in_data_port_id + "_" + key
            
            # If a new port-specific key is encountered, initialise a list.
            # The list is initially named identically to this key.
            if not ikey in self.ikeys_to_dkeys:
                if count_ikey_new < SS.MAX_ALERTS_IKEY_NEW:
                    log.info("%s - DataStorage is newly encountering data "
                             "from a DataPort with key '%s'."
                             % (Timestamp(), ikey))
                # TODO: Set up a safety mode where key is a distinct ikey.
                self.ikeys_to_dkeys[ikey] = key
                count_ikey_new += 1
            
            dkey = self.ikeys_to_dkeys[ikey]
            
            if not dkey in self.data:
                if count_dkey_new < SS.MAX_ALERTS_DKEY_NEW:
                    log.info("%s - DataStorage has begun storing data "
                             "in a list with key '%s'." 
                             % (Timestamp(), dkey))
                self.data[dkey] = [None]*len(self.timestamps)
                
                # The first element in a list determines its data type.
                self.data_types[dkey] = infer_data_type(element)
                count_dkey_new += 1
            
            # Add the new element to the list with str-to-type conversion.
            # Note the function call.
            try:
                # TODO: Some data types do not convert, e.g. NoneType. Consider how to fix/avoid.
                self.data[dkey][-1] = self.data_types[dkey](element)
            except Exception as e:
                # TODO: Handle changes in data type for messy datasets.
                raise e
                
        if count_ikey_new > SS.MAX_ALERTS_IKEY_NEW:
            log.info("%s - In total, DataStorage has newly encountered data "
                     "from a DataPort with %i unseen keys."
                     % (Timestamp(), count_ikey_new))
        if count_dkey_new > SS.MAX_ALERTS_DKEY_NEW:
            log.info("%s - In total, DataStorage has begun storing data "
                     "in %i new keyed lists."
                     % (Timestamp(), count_dkey_new))
        
        # Flick a switch so that learners can start ingesting new data.
        # Note: Resolving awaited futures are priority microtasks.
        # The following reset runs after the learners are signalled.
        self.has_new_data.set_result(True)
        self.has_new_data = asyncio.Future()
        
    def info(self):
        """
        Utility method to give user info about data ports and storage.
        """
        if len(self.data_types.keys()) > SS.MAX_INFO_KEYS_EXAMPLE:
            data_types_keys = list(self.data_types.keys())
            len_start = int(np.ceil(SS.MAX_INFO_KEYS_EXAMPLE/2))
            len_end = SS.MAX_INFO_KEYS_EXAMPLE - len_start
            example_keys = ", ".join(key + " (" + self.data_types[key].__name__ + ")"
                                     for key in data_types_keys[:len_start])
            example_keys += ", ..., "
            example_keys += ", ".join(key + " (" + self.data_types[key].__name__ + ")"
                                      for key in data_types_keys[-len_end:])
        else:
            example_keys = ", ".join(key + " (" + self.data_types[key].__name__ + ")"
                                     for key in self.data_types)
            
        if len(self.ikeys_to_dkeys) > SS.MAX_INFO_PIPE_EXAMPLE:
            ikeys_to_dkeys_keys = list(self.ikeys_to_dkeys.keys())
            len_start = int(np.ceil(SS.MAX_INFO_PIPE_EXAMPLE/2))
            len_end = SS.MAX_INFO_PIPE_EXAMPLE - len_start
            example_pipe = ", ".join("{" + key + " -> " + self.ikeys_to_dkeys[key] + "}"
                                     for key in ikeys_to_dkeys_keys[:len_start])
            example_pipe += ", ..., "
            example_pipe += ", ".join("{" + key + " -> " + self.ikeys_to_dkeys[key] + "}"
                                      for key in ikeys_to_dkeys_keys[-len_end:])
        else:
            example_pipe = ", ".join("{" + key + " -> " + self.ikeys_to_dkeys[key] + "}"
                                     for key in self.ikeys_to_dkeys)

        
        log.info("Stored data is arranged into lists identified as follows.")
        log.info("Keys: %s" % example_keys)
        log.info("DataPorts pipe data into the lists as follows.")
        log.info("Pipe: %s" % example_pipe)
        
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
            
            if not key_port in self.ikeys_to_dkeys:
                log.warning("%s - No existing DataPort keys data with '%s'. "
                            "Ignoring update request: {%s -> %s}"
                            % (Timestamp(), key_port, key_port, key_storage))
            else:
                key_storage_old = self.ikeys_to_dkeys[key_port]
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
                        self.ikeys_to_dkeys[key_port] = key_storage
                        
                # Handle what is effectively the renaming of a list.
                else:
                    self.data_types[key_storage] = self.data_types.pop(key_storage_old)
                    self.data[key_storage] = self.data.pop(key_storage_old)
                    
                    # Update directions for the DataPort.
                    self.ikeys_to_dkeys[key_port] = key_storage
                
        
        
    def get_dataframe(self):
        """
        A utility method converting data dictionary into a Pandas dataframe.
        This is slow and should be called sparingly.
        """
        df = pd.DataFrame.from_dict(self.data)
        df.index = self.timestamps
        return df