# -*- coding: utf-8 -*-
"""
Created on Fri May 12 22:21:05 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .settings import SystemSettings as SS
from .concurrency import create_async_task_from_sync
from .data import (DataType, DataFormatX, DataFormatY, reformat_x, reformat_y)

import asyncio
import ast
import random
from copy import copy, deepcopy

import numpy as np

# TODO: Redesign so the inference/conversion is done at DataPort interface?
#       This will allow CSV text file inputs to be treated differently from other inputs.
def infer_data_type(in_element):
    element_type = type(in_element)
    
    # Check if a string data type can be converted to something else.
    if element_type == str:
        try:
            element_type = type(ast.literal_eval(in_element))
        except:
            pass

    if element_type == float:
        data_type = DataType.FLOAT
    elif element_type == int:
        data_type = DataType.INTEGER
    else:
        data_type = DataType.CATEGORICAL
    
    return data_type



class DataCollection:
    def __init__(self):
        self.timestamps = list()
        self.data = dict()

    def __add__(self, other):
        """
        Creates and returns a new DataCollection that is merged from this and another.
        Does no error checking on the structure.
        """

        collection = DataCollection()

        collection.timestamps = self.timestamps + other.timestamps
        for key in self.data:
            collection.data[key] = self.data[key] + other.data[key]

        return collection

    # TODO: Make sure in_idx_start is less than in_idx_stop when modulo operation is applied.
    def split_by_range(self, in_idx_start: int = 0, in_idx_stop: int = None):
        """
        Return DataCollections with instances inside/outside the specified range.
        Note: Creates shallow copies.
        """
        if in_idx_start == 0 and ((in_idx_stop is None) or (in_idx_stop > self.get_amount())):
            collection_in = copy(self)
            collection_out = DataCollection()
            collection_out.data = {key: list() for key in self.data}
        else:
            collection_in = DataCollection()
            collection_in.timestamps = self.timestamps[in_idx_start:in_idx_stop]
            collection_in.data = {key: self.data[key][in_idx_start:in_idx_stop] for key in self.data}
            
            collection_out = DataCollection()
            collection_out.timestamps = self.timestamps[:in_idx_start] + self.timestamps[in_idx_stop:]
            collection_out.data = {key: (self.data[key][:in_idx_start] + self.data[key][in_idx_stop:]) 
                                for key in self.data}

        return collection_in, collection_out
    
    def split_by_content(self, in_key: str, in_value):
        """
        Return DataCollections with instances including/excluding a specified key-value pair.
        Note: Creates shallow copies.
        """
        indices_in = list()
        indices_out = list()

        for idx, val in enumerate(self.data[in_key]):
            if val == in_value:
                indices_in.append(idx)
            else:
                indices_out.append(idx)

        collection_in = DataCollection()
        collection_in.timestamps = [self.timestamps[idx] for idx in indices_in]
        collection_in.data = {key: [self.data[key][idx] for idx in indices_in] for key in self.data}
        
        collection_out = DataCollection()
        collection_out.timestamps = [self.timestamps[idx] for idx in indices_out]
        collection_out.data = {key: [self.data[key][idx] for idx in indices_out] for key in self.data}

        return collection_in, collection_out
    
    # TODO: Consider cases where empty collections are returned. Where to catch exceptions?
    def split_by_fraction(self, in_fraction: float = 0.25, in_seed: int = 0):
        """
        Return DataCollections with/without randomly selected instances.
        The instances constitute a fraction of the collected data.
        Note: Creates shallow copies.
        """
        n_instances = self.get_amount()
        random.seed(in_seed)
        list_indices = random.sample(range(n_instances), n_instances)

        # Ensure the number of indices selected is appropriately bounded.
        idx_in_stop = min(max(0, int(in_fraction * n_instances)), 
                          n_instances)
        idx_out_start = idx_in_stop - n_instances
        list_indices_in = list_indices[:idx_in_stop]
        list_indices_out = list_indices[idx_out_start:]

        collection_in = DataCollection()
        collection_in.timestamps = [self.timestamps[idx] for idx in list_indices_in]
        collection_in.data = {key: [self.data[key][idx] for idx in list_indices_in] 
                              for key in self.data}
        
        collection_out = DataCollection()
        collection_out.timestamps = [self.timestamps[idx] for idx in list_indices_out]
        collection_out.data = {key: [self.data[key][idx] for idx in list_indices_out] 
                               for key in self.data}

        return collection_in, collection_out

    def get_data(self, in_keys_features, in_key_target: str,
                 in_format_x: DataFormatX = None, in_format_y: DataFormatY = None,
                 in_fraction: float = 1.0):
        """
        Return data as a set of features x and a target y.
        """

        source = self.data

        # Copy out the required data in default DataFormatX and DataFormatY style.
        x = deepcopy({key_feature:source[key_feature] for key_feature in in_keys_features})
        y = deepcopy(source[in_key_target])

        # Randomly sample a fraction of the data if desired.
        if in_fraction < 1:
            amount_selected = self.get_amount()
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
    
    def get_amount(self, in_idx_start: int = 0, in_idx_stop: int = None):
        return len(self.timestamps[in_idx_start:in_idx_stop])



# TODO: Consider how the data is best stored, including preallocated arrays.
# TODO: Update comments.
class DataStorage:
    """
    A container that manages collections of data used for machine learning processes.
    """
    
    def __init__(self):
        log.info("%s - Initialising DataStorage." % Timestamp())

        # self.observations = DataCollection()
        # self.queries = DataCollection()

        # Store observations/queries as dicts of data collections keyed by user-provided tags.
        # If no tags are provided, data will be keyed to a no-tag ID.
        self.collection_id_no_tag = 0
        self.collection_id_new = 1
        self.observations = dict()
        self.queries = dict()
        # self.observations = {self.collection_id_no_tag: DataCollection()}
        # self.queries = {self.collection_id_no_tag: DataCollection()}
        self.tag_to_collection_ids = dict()

        self.data_types = dict()    # The data types for each keyed list.
        
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

    def get_key_dict(self):
        """
        The data-types dictionary contains keys for data in all stored collections.
        """
        return self.data_types
    
    def get_tag_values(self, in_tag):
        return list(self.tag_to_collection_ids[in_tag].keys())

    def get_observations(self, in_tags = None):
        return self.observations[self.get_collection_id(in_tags)]
    
    def get_queries(self, in_tags = None):
        return self.queries[self.get_collection_id(in_tags)]
    
    def get_amount(self, from_queries = False):
        if not from_queries:
            source = self.observations
        else:
            source = self.queries

        amount = 0
        for collection in source.values():
            amount += collection.get_amount()
        return amount
    
    def get_collection(self, in_tags_inclusive = None, in_tags_exclusive = None):
        """
        If there are no inclusive tags, all collections are selected.
        """
        if in_tags_inclusive is None:
            in_tags_inclusive = dict()
        if in_tags_exclusive is None:
            in_tags_exclusive = dict()
        
        if len(in_tags_inclusive) == 0:
            set_collection_ids = set(self.observations.keys())
        else:
            set_collection_ids = set()

        for key_tag, value_tag in in_tags_inclusive.items():
            set_collection_ids = set_collection_ids | self.tag_to_collection_ids[key_tag][value_tag]
        for key_tag, value_tag in in_tags_exclusive.items():
            set_collection_ids = set_collection_ids - self.tag_to_collection_ids[key_tag][value_tag]

        collection = None
        for collection_id in list(set_collection_ids):
            if collection is None:
                collection = self.observations[collection_id]
            else:
                collection = collection + self.observations[collection_id]

        return collection


    # TODO: Catch exceptions.
    def get_collection_id(self, in_tags = None):
    
        if in_tags is None:
            collection_id = self.collection_id_no_tag
        else:
            set_collection_ids = None
            for key_tag, value_tag in in_tags.items():
                if set_collection_ids is None:
                    set_collection_ids = self.tag_to_collection_ids[key_tag][value_tag]
                else:
                    set_collection_ids = set_collection_ids & self.tag_to_collection_ids[key_tag][value_tag]

            collection_id = list(set_collection_ids)[0]

        return collection_id

    # TODO: Update info logging once terminology is settled. Fix logging for tags.
    def store_data(self, in_timestamp: Timestamp, in_data_port_id, in_keys, in_elements, in_data_types, in_tags: dict,
                   as_query: bool = False):

        if in_tags is None:
            if not self.collection_id_no_tag in self.observations:
                self.observations[self.collection_id_no_tag] = DataCollection()
                self.queries[self.collection_id_no_tag] = DataCollection()
            collection_id = self.collection_id_no_tag
        else:
            set_collection_ids = None
            for key_tag, value_tag in in_tags.items():
                if not key_tag in self.tag_to_collection_ids:
                    self.tag_to_collection_ids[key_tag] = dict()
                if not value_tag in self.tag_to_collection_ids[key_tag]:
                    self.tag_to_collection_ids[key_tag][value_tag] = set()

                # Search the intersection of all collection ids associated with tags.
                # There should only be one collection per combination of tags.
                if set_collection_ids is None:
                    set_collection_ids = self.tag_to_collection_ids[key_tag][value_tag]
                else:
                    set_collection_ids = set_collection_ids & self.tag_to_collection_ids[key_tag][value_tag]

            # If no collection can be found, create a new one and associate it with relevant tags.
            if len(set_collection_ids) == 0:
                self.observations[self.collection_id_new] = DataCollection()
                self.queries[self.collection_id_new] = DataCollection()
                for key_tag, value_tag in in_tags.items():
                    self.tag_to_collection_ids[key_tag][value_tag].add(self.collection_id_new)
                collection_id = self.collection_id_new
                self.collection_id_new += 1
            else:
                collection_id = list(set_collection_ids)[0]

        if as_query:
            timestamps = self.queries[collection_id].timestamps
            dict_storage = self.queries[collection_id].data
        else:
            timestamps = self.observations[collection_id].timestamps
            dict_storage = self.observations[collection_id].data
            
        timestamps.append(in_timestamp)

        # Extend all existing data or query lists by one empty slot.
        for dkey in dict_storage:
            dict_storage[dkey].append(None)
        
        count_ikey_new = 0
        count_dkey_new = 0
        for key, element, data_type in zip(in_keys, in_elements, in_data_types):
            
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
            if not dkey in self.observations[collection_id].data:
                if count_dkey_new < SS.MAX_ALERTS_DKEY_NEW:
                    log.info("%s   DataStorage has begun storing elements of "
                             "data/queries in a list with key '%s'." 
                             % (Timestamp(None), dkey))
                self.observations[collection_id].data[dkey] = [None]*len(self.observations[collection_id].timestamps)
                self.queries[collection_id].data[dkey] = [None]*len(self.queries[collection_id].timestamps)
                
                # If the type has not been determined elsewhere, the first element decides.
                # TODO: Improve inference process. Maybe update as new data is encountered.
                if data_type is None:
                    self.data_types[dkey] = infer_data_type(element)
                else:
                    self.data_types[dkey] = data_type
                count_dkey_new += 1
            
            # Add the new element to the list with str-to-type conversion.
            try:
                # TODO: Some data types do not convert, e.g. NoneType. Consider how to fix/avoid.
                dict_storage[dkey][-1] = self.data_types[dkey].convert(element)
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
            self.has_new_queries.set_result(in_tags)
            self.has_new_queries = asyncio.Future()
        else:
            self.has_new_observations.set_result(in_tags)
            self.has_new_observations = asyncio.Future()


    # def get_unique_values(self, in_key, do_check_category = True, from_queries = False):
        
    #     if do_check_category and not self.data_types[in_key] == DataType.CATEGORICAL:
    #         text_error = ("Attempting to acquire unique values from "
    #                         "a list of non-categorical data type.")
    #         log.error("%s - %s" % (Timestamp(), text_error))
    #         raise Exception(text_error)
        
    #     if from_queries:
    #         unique_values = list(dict.fromkeys(self.queries.data[in_key]))
    #     else:
    #         unique_values = list(dict.fromkeys(self.observations.data[in_key]))

    #     return unique_values
        

    def info(self):
        """
        Utility method to give user info about data ports and storage.
        """
        if len(self.data_types.keys()) > SS.MAX_INFO_KEYS_EXAMPLE:
            data_types_keys = list(self.data_types.keys())
            len_start = int(np.ceil(SS.MAX_INFO_KEYS_EXAMPLE/2))
            len_end = SS.MAX_INFO_KEYS_EXAMPLE - len_start
            example_keys = ", ".join(key + " (" + self.data_types[key].to_string() + ")"
                                     for key in data_types_keys[:len_start])
            example_keys += ", ..., "
            example_keys += ", ".join(key + " (" + self.data_types[key].to_string() + ")"
                                      for key in data_types_keys[-len_end:])
        else:
            example_keys = ", ".join(key + " (" + self.data_types[key].to_string() + ")"
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
    
    # # TODO: Error-check indices.
    # def get_data(self, in_keys_features, in_key_target, 
    #              in_format_x = None, in_format_y = None,
    #             #  in_idx_start = 0, in_idx_stop = None,
    #              from_queries = False):

    #     source = self.observations
    #     if from_queries:
    #         source = self.queries

    #     return source.get_data(in_keys_features = in_keys_features,
    #                            in_key_target = in_key_target,
    #                            in_format_x = in_format_x,
    #                            in_format_y = in_format_y)

            
        # # Copy out the required data in default DataFormatX and DataFormatY style.
        # x = {key_feature:deepcopy(source[key_feature][in_idx_start:in_idx_stop]) 
        #      for key_feature in in_keys_features}
        # y = deepcopy(source[in_key_target][in_idx_start:in_idx_stop])

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