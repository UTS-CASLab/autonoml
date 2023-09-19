# -*- coding: utf-8 -*-
"""
Created on Fri May 12 22:21:05 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .settings import SystemSettings as SS
from .concurrency import create_async_task_from_sync
from .data import DataFormatX, DataFormatY, reformat_x, reformat_y

import asyncio
import ast
import random
from copy import deepcopy

import numpy as np

# TODO: Redesign so the inference/conversion is done at DataPort interface?
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



class DataCollection:
    def __init__(self):
        self.timestamps = list()
        self.data = dict()

    def get_data(self, in_keys_features, in_key_target: str,
                 in_format_x: DataFormatX = None, in_format_y: DataFormatY = None,
                 in_idx_start: int = 0, in_idx_end: int = None,
                 in_fraction: float = 1):

        source = self.data
            
        # Copy out the required data in default DataFormatX and DataFormatY style.
        x = {key_feature:deepcopy(source[key_feature][in_idx_start:in_idx_end]) 
                for key_feature in in_keys_features}
        y = deepcopy(source[in_key_target][in_idx_start:in_idx_end])

        # Randomly sample a fraction of the data if desired.
        if in_fraction < 1:
            amount_selected = self.get_amount(in_idx_start, in_idx_end)
            size_sample = max(1, int(in_fraction * amount_selected))
            random.seed(0)
            idx_list = random.sample(range(amount_selected), size_sample)

            x = {key:[x[key][idx] for idx in idx_list] for key in x}
            y = [y[idx] for idx in idx_list]

        if in_format_x is None:
            in_format_x = DataFormatX(0)
        if in_format_y is None:
            in_format_y = DataFormatY(0)

        # Reformat the data.
        # If formats were not specified, the data is retrieved in 'standard' format.
        x = reformat_x(in_data = x, 
                        in_format_old = DataFormatX(0),
                        in_format_new = in_format_x,
                        in_keys_features = in_keys_features)
        y = reformat_y(in_data = y, 
                        in_format_old = DataFormatY(0),
                        in_format_new = in_format_y)

        return x, y
    
    def get_amount(self, in_idx_start: int = 0, in_idx_end: int = None):
        return len(self.timestamps[in_idx_start:in_idx_end])



# TODO: Consider how the data is best stored, including preallocated arrays.
# TODO: Update comments.
class DataStorage:
    """
    A container that manages collections of data used for machine learning processes.
    """
    
    def __init__(self):
        log.info("%s - Initialising DataStorage." % Timestamp())
        
        self.observations = DataCollection()
        self.queries = DataCollection()

        # self.timestamps_observations = list()
        # self.observations = dict()          # Stored data arranged in keyed lists.
        
        # self.timestamps_queries = list()
        # self.queries = dict()       # Stored queries arranged in keyed lists.
        
        self.data_types = dict()    # The data types for each keyed list.
        # Note: Data types are actual types not strings.
        
        # Ingested data arrives from data ports.
        # For data port X, this data is sent as a list of elements.
        # The elements have keys: X_0, X_1, etc.
        # Define a dict that links port-specific keys to storage-specific keys.
        # This directs elements of incoming data to the right list.
        self.ikeys_to_dkeys = dict()
        
        # Set up variables that can be awaited elsewhere.
        # These 'switches', when flicked, signals learners to ingest new data.
        self.has_new_observations = None
        self.has_new_queries = None
        # Note: These futures must be instantiated on the right loop, i.e. only within a coroutine.

        create_async_task_from_sync(self.prepare)

    async def prepare(self):
        # Instantiate the futures now that this code is running internally within an event loop.
        self.has_new_observations = asyncio.Future()
        self.has_new_queries = asyncio.Future()
    
    # # Add a key to both the data and query storage with associated data type.
    # # Fill associated lists thus far with None.
    # def add_data_key(self, in_key, in_type):
        
    #     if not in_key in self.observations:
    #         self.observations[in_key] = [None]*len(self.timestamps)
    #     if not in_key in self.queries:
    #         self.queries[in_key] = [None]*len(self.timestamps)
        
    # TODO: Update info logging once terminology is settled.
    def store_data(self, in_timestamp, in_data_port_id, in_keys, in_elements, as_query = False):

        if as_query:
            timestamps = self.queries.timestamps
            dict_storage = self.queries.data
        else:
            timestamps = self.observations.timestamps
            dict_storage = self.observations.data
            
        timestamps.append(in_timestamp)
        
        # Extend all existing data or query lists by one empty slot.
        for dkey in dict_storage:
            dict_storage[dkey].append(None)
        
        count_ikey_new = 0
        count_dkey_new = 0
        for key, element in zip(in_keys, in_elements):
            
            ikey = in_data_port_id + "_" + key
            
            # If a new port-specific key is encountered, initialise a list.
            # The list is initially named identically to this key.
            if not ikey in self.ikeys_to_dkeys:
                if count_ikey_new < SS.MAX_ALERTS_IKEY_NEW:
                    log.info("%s   DataStorage is newly encountering elements "
                             "of data/queries from a DataPort with key '%s'."
                             % (Timestamp(None), ikey))
                # TODO: Set up a safety mode where key is a distinct ikey.
                self.ikeys_to_dkeys[ikey] = key
                count_ikey_new += 1
            
            dkey = self.ikeys_to_dkeys[ikey]
            
            # Both data/queries must have the same keys.
            if not dkey in self.observations.data:
                if count_dkey_new < SS.MAX_ALERTS_DKEY_NEW:
                    log.info("%s   DataStorage has begun storing elements of "
                             "data/queries in a list with key '%s'." 
                             % (Timestamp(None), dkey))
                self.observations.data[dkey] = [None]*len(self.observations.timestamps)
                self.queries.data[dkey] = [None]*len(self.queries.timestamps)
                
                # The first element in a list determines its data type.
                self.data_types[dkey] = infer_data_type(element)
                count_dkey_new += 1
            
            # Add the new element to the list with str-to-type conversion.
            # Note the function call.
            try:
                # TODO: Some data types do not convert, e.g. NoneType. Consider how to fix/avoid.
                dict_storage[dkey][-1] = self.data_types[dkey](element)
            except Exception as e:
                # TODO: Handle changes in data type for messy datasets.
                raise e
                
        if count_ikey_new > SS.MAX_ALERTS_IKEY_NEW:
            log.info("%s   In total, DataStorage has newly encountered data/queries "
                     "from a DataPort with %i unseen keys."
                     % (Timestamp(None), count_ikey_new))
        if count_dkey_new > SS.MAX_ALERTS_DKEY_NEW:
            log.info("%s   In total, DataStorage has begun storing data/queries "
                     "in %i new keyed lists."
                     % (Timestamp(None), count_dkey_new))
        
        # Flick a switch so that learners can start ingesting new data.
        # Note: Resolving awaited futures are priority microtasks.
        # The following reset runs after the learners are signalled.
        if as_query:
            self.has_new_queries.set_result(True)
            self.has_new_queries = asyncio.Future()
        else:
            self.has_new_observations.set_result(True)
            self.has_new_observations = asyncio.Future()
        
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
        
    # # TODO: This currently breaks a ProblemSolver with feature keys. Decide what do.
    # # TODO: Update for queries!
    # def update(self, in_keys_port, in_keys_storage):
    #     """
    #     Take in a list of DataPort data keys.
    #     Update the DataStorage lists into which the data is directed.
    #     This may be as simple as renaming lists or may involve merging.
    #     Note: This process also shifts historically stored data across.
    #     """
        
    #     for count_key in range(len(in_keys_port)):
    #         key_port = in_keys_port[count_key]
    #         key_storage = in_keys_storage[count_key]
            
    #         if not key_port in self.ikeys_to_dkeys:
    #             log.warning("%s - No existing DataPort keys data with '%s'. "
    #                         "Ignoring update request: {%s -> %s}"
    #                         % (Timestamp(), key_port, key_port, key_storage))
    #         else:
    #             key_storage_old = self.ikeys_to_dkeys[key_port]
    #             data_type_old = self.data_types[key_storage_old]
                
    #             # Handle merging with pre-existing lists.
    #             if key_storage in self.observations:
    #                 data_type_existing = self.data_types[key_storage]
                    
    #                 # Refuse for clashing data types.
    #                 # TODO: Consider allowing it but generalising data types.
    #                 if not data_type_existing == data_type_old:
    #                     log.warning("%s - DataStorage already contains list '%s' with type '%s'. "
    #                                 "Refusing to merge in list '%s' with type '%s'."
    #                                 % (Timestamp(), key_storage, data_type_existing.__name__, 
    #                                    key_storage_old, data_type_old.__name__))
    #                 else:
    #                     log.warning("%s - DataStorage already contains list '%s'. "
    #                                 "Proceeding to overwrite with list '%s' where values exist."
    #                                 % (Timestamp(), key_storage, key_storage_old))
    #                     self.data_types.pop(key_storage_old)
    #                     list_old = self.observations.pop(key_storage_old)
    #                     for count_element in range(len(list_old)):
    #                         element = list_old[count_element]
    #                         if element is not None:
    #                             self.observations[key_storage][count_element] = element
                        
    #                     # Update directions for the DataPort.
    #                     self.ikeys_to_dkeys[key_port] = key_storage
                        
    #             # Handle what is effectively the renaming of a list.
    #             else:
    #                 self.data_types[key_storage] = self.data_types.pop(key_storage_old)
    #                 self.observations[key_storage] = self.observations.pop(key_storage_old)
                    
    #                 # Update directions for the DataPort.
    #                 self.ikeys_to_dkeys[key_port] = key_storage
    
    # TODO: Error-check indices.
    def get_data(self, in_keys_features, in_key_target, 
                 in_format_x = None, in_format_y = None,
                 in_idx_start = 0, in_idx_end = None,
                 from_queries = False):

        source = self.observations
        if from_queries:
            source = self.queries

        return source.get_data(in_keys_features = in_keys_features,
                               in_key_target = in_key_target,
                               in_format_x = in_format_x,
                               in_format_y = in_format_y,
                               in_idx_start = in_idx_start,
                               in_idx_end = in_idx_end)

            
        # # Copy out the required data in default DataFormatX and DataFormatY style.
        # x = {key_feature:deepcopy(source[key_feature][in_idx_start:in_idx_end]) 
        #      for key_feature in in_keys_features}
        # y = deepcopy(source[in_key_target][in_idx_start:in_idx_end])

        # if in_format_x is None:
        #     in_format_x = DataFormatX(0)
        # if in_format_y is None:
        #     in_format_y = DataFormatY(0)

        # # Reformat the data.
        # # If formats were not specified, the data is retrieved in 'standard' format.
        # x = reformat_x(in_data = x, 
        #                in_format_old = DataFormatX(0),
        #                in_format_new = in_format_x,
        #                in_keys_features = in_keys_features)
        # y = reformat_y(in_data = y, 
        #                in_format_old = DataFormatY(0),
        #                in_format_new = in_format_y)

        # return x, y
        
    # def get_dataframe(self, from_queries = False):
    #     """
    #     A utility method converting data dictionary into a Pandas dataframe.
    #     This is slow and should be called sparingly.
    #     """
    #     source = self.observations
    #     timestamps = self.timestamps_observations
    #     if from_queries:
    #         source = self.queries
    #         timestamps = self.timestamps_queries

    #     df = pd.DataFrame.from_dict(source)
    #     df.index = timestamps
    #     return df