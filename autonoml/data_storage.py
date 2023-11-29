# -*- coding: utf-8 -*-
"""
Created on Fri May 12 22:21:05 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .concurrency import create_async_task_from_sync

from typing import Type, Union, List, Dict, Any
import bisect

import asyncio
from copy import deepcopy

import __main__
import os
import pyarrow as pa
import pyarrow.ipc as ipc

import numpy as np



class DataCollectionBase:
    """
    A base container for collections of data.
    Each base collection stores IDs and insertion timestamps corresponding to instances of data.
    Subclasses store the actual data.
    Note: The uniqueness of IDs is managed externally, i.e. by a DataStorage object.
    """
    def __init__(self, in_ids: List[int] = None, in_timestamps: List[Timestamp] = None, *args, **kwargs):
        self.ids = list()
        self.timestamps = list()
        if not in_ids is None:
            self.ids = in_ids
        if not in_timestamps is None:
            self.timestamps = in_timestamps

    def extend(self, in_ids: List[int], in_timestamps: List[Timestamp]):
        self.ids.extend(in_ids)
        self.timestamps.extend(in_timestamps)

    def get_index_after_id(self, in_id: int):
        """
        If the ID list is sorted, return the first index of a value greater than a specified ID.
        Warning: Do not use for unsorted lists, as the fast bisect algorithm is used.
        """
        return bisect.bisect_right(self.ids, in_id)

    def get_index_after_timestamp(self, in_timestamp: Timestamp):
        """
        If the timestamp list is sorted, return the first index of a value greater than a specified timestamp.
        Warning: Do not use for unsorted lists, as the fast bisect algorithm is used.
        """
        return bisect.bisect_right(self.timestamps, in_timestamp)

    def get_amount(self):
        return len(self.ids)


class DataCollection(DataCollectionBase):
    """
    The standard container for a collection of data.
    All data should be kept in this format unless temporarily reformatted for processing.
    """

    def __init__(self, in_data: pa.Table = None, 
                 in_ids: List[int] = None, in_timestamps: List[Timestamp] = None, *args, **kwargs):
        """
        If IDs/timestamps exist, directly attach them with data to a DataCollection object.
        If IDs/timestamps do not exist, insert data instead and generate new IDs/timestamps.
        Note: Avoid initialising DataCollection directly if ID uniqueness matters.
        """
        super().__init__(in_ids = in_ids, in_timestamps = in_timestamps, *args, **kwargs)

        empty_schema = pa.schema([])
        self.data = pa.Table.from_batches([], schema=empty_schema)
        if not in_data is None:
            if in_ids is None and in_timestamps is None:
                self.insert(in_data)
            else:
                self.data = in_data

    def insert(self, in_data: pa.Table, in_id_last: int = -1):
        """
        Insert data and extend associated list of IDs with increments from a specified ID.
        Return the last ID to help in maintaining uniqueness.
        Also store insertion timestamp.
        """
        # TODO: Consider cases where users do not want permissive type updating.
        # TODO: Consider cases where type updating can actually break models, e.g. classifiers.
        self.data = pa.concat_tables([self.data, in_data], promote_options = "permissive")
        self.extend(in_ids = [in_id_last + i for i in range(1, 1 + in_data.num_rows)],
                    in_timestamps = [Timestamp()]*in_data.num_rows)
        return in_id_last + in_data.num_rows

    def split_by_special_range(self, in_id_start_exclusive: int = None, 
                               in_id_stop_inclusive: int = None):
        """
        Take a start ID and a stop ID and convert this into a range of indices for the data.
        Non-traditionally, the range starts after the start ID and stops after the stop ID.
        Return DataCollection objects for data inside and outside that range.
        """
        
        id_start = self.ids[0] - 1 if in_id_start_exclusive is None else in_id_start_exclusive
        id_stop = self.ids[-1] + 1 if in_id_stop_inclusive is None else in_id_stop_inclusive

        indices = np.arange(self.get_amount())
        idx_start = self.get_index_after_id(id_start)
        idx_stop = self.get_index_after_id(id_stop)

        collection_in = DataCollection(in_data = self.data.take(indices[idx_start:idx_stop]),
                                       in_ids = self.ids[idx_start:idx_stop],
                                       in_timestamps = self.timestamps[idx_start:idx_stop])
        indices_out = np.concatenate((indices[:idx_start], indices[idx_stop:]))
        collection_out = DataCollection(in_data = self.data.take(indices_out), 
                                        in_ids = self.ids[:idx_start] + self.ids[idx_stop:],
                                        in_timestamps = self.timestamps[:idx_start] 
                                                        + self.timestamps[idx_stop:])

        return collection_in, collection_out
        
    
    def prepare_xy(self, in_keys_features: List[str], in_key_target: str):
        """
        Return a special version of DataCollection prepared with features (x) and target (y).
        """
        x = self.data.select(in_keys_features)
        y = self.data.column(in_key_target)

        return DataCollectionXY(in_x = x, in_y = y, 
                                in_ids = self.ids, in_timestamps = self.timestamps)
    
    def quick_split_xy(self):
        """
        A convenience function that quickly converts to DataCollectionXY.
        Assumes the last column array in the data table is y and all else is x.
        Loses y header.
        """
        idx_col = self.data.num_columns - 1
        y = self.data.column(idx_col)
        x = self.data.remove_column(idx_col)

        return DataCollectionXY(in_x = x, in_y = y, 
                                in_ids = self.ids, in_timestamps = self.timestamps)
    
    def combine_chunks(self):
        """
        If data is a chunked table, combine those chunks.
        Necessary when writing to csv file, otherwise each chunk has its own schema.
        """
        self.data = self.data.combine_chunks()


class DataCollectionXY(DataCollectionBase):
    """
    A container for data where feature (x) and target (y) selection has already been done.
    Is primarily used for HPO where repeat data manipulations are computationally expensive.
    It is not intended for this container to grow, only to be sliced.
    """
    def __init__(self, in_x: pa.Table, in_y: pa.ChunkedArray,
                 in_ids: List[int] = None, in_timestamps: List[Timestamp] = None, *args, **kwargs):
        super().__init__(in_ids = in_ids, in_timestamps = in_timestamps, *args, **kwargs)
        self.x = in_x
        self.y = in_y
    
    def get_data(self, in_fraction: float = 1.0):
        """
        Returns a specified fractional slice of the data.
        Note: This assumes the data is already randomly shuffled, if applicable.
        """
        if in_fraction == 1.0:
            x, y = self.x, self.y
        else:
            n_instances = self.get_amount()
            n_samples = min(max(0, int(n_instances * in_fraction)), n_instances)
            indices = np.arange(n_instances)
            x = self.x.take(indices[:n_samples])
            y = self.y.take(indices[:n_samples])

        return x, y

    def split_randomly_by_fraction(self, in_fraction: float = 0.25, in_seed: int = 0):
        """
        Return shuffled DataCollectionXYs inside/outside a specified fraction of instances.
        Optionally apply shuffling to IDs and timestamps or just ignore it altogether for efficency.
        """
        np.random.seed(in_seed)
        n_instances = self.get_amount()
        n_samples = min(max(0, int(n_instances * in_fraction)), n_instances)
        indices = np.random.permutation(np.arange(n_instances))

        # Store shuffled IDs and timestamps as unsorted lists, but beware of constrained functionality.
        collection_in = DataCollectionXY(in_x = self.x.take(indices[:n_samples]), 
                                            in_y = self.y.take(indices[:n_samples]),
                                            in_ids = [self.ids[idx] for idx in indices[:n_samples]],
                                            in_timestamps = [self.timestamps[idx] for idx in indices[:n_samples]])
        collection_out = DataCollectionXY(in_x = self.x.take(indices[n_samples:]),
                                            in_y = self.y.take(indices[n_samples:]),
                                            in_ids = [self.ids[idx] for idx in indices[n_samples:]],
                                            in_timestamps = [self.timestamps[idx] for idx in indices[n_samples:]])

        return collection_in, collection_out
    
    def quick_merge_xy(self):
        """
        A convenience function that converts to DataCollection.
        Concatenates y to x as the final column of a data table.
        Fakes y header.
        """
        return DataCollection(in_data = self.x.append_column("_", self.y), 
                              in_ids = self.ids, in_timestamps = self.timestamps)
    


# TODO: Consider robustness for IDs.
class SharedMemoryManager:
    """
    Used in multiprocessing for efficiently sharing data between processes.
    It does so by writing to and reading from memory-mapped files on disk.
    When multiprocessing is disabled, simply stores references to data used for model development.

    Warning: Does not presently preserve IDs/timestamps in the case of multiprocessing.
    """
    
    count = 0

    def __init__(self, in_uses: int = 1, do_mp: bool = False):

        self.name = "shared_" + str(SharedMemoryManager.count)
        SharedMemoryManager.count += 1
        self.prefix = "./temp/"
        os.makedirs(self.prefix, exist_ok = True)
        self.n_sets = 0

        # Set up a counter to note how many objects are using or about to use associated data.
        # Once it goes to zero, there should be no more references to memory-mapped data.
        # The local-disk files will be deleted at that stage.
        self.uses = in_uses

        self.do_mp = do_mp
        self.observations = None
        self.sets_training = None
        self.sets_validation = None

    @staticmethod
    def get_size(in_collection: DataCollection):
        """
        Mock up a lightweight stream to count how many bytes writing a table will take.
        """
        sink = pa.MockOutputStream()
        with pa.ipc.new_stream(sink, in_collection.data.schema) as writer:
            writer.write_table(in_collection.data)
        return sink.size()

    @staticmethod
    def save(in_collection: DataCollectionXY, in_filepath: str):

        collection = in_collection.quick_merge_xy()
        with pa.memory_map(in_filepath, "w") as sink:
            sink.resize(SharedMemoryManager.get_size(collection))
            with ipc.RecordBatchStreamWriter(sink, collection.data.schema) as writer:
                writer.write_table(collection.data)

    # TODO: Decide what the best choice for multiprocessing is.
    @staticmethod
    def load(in_filepath: str):

        # with pa.OSFile(in_filepath, "r") as source:
        with pa.memory_map(in_filepath, "r") as source:
            with ipc.RecordBatchStreamReader(source) as reader:
                data = reader.read_all()
                collection = DataCollection(data).quick_split_xy()

        return collection

    def save_observations(self, in_observations: DataCollectionXY, 
                          in_sets_training: List[DataCollectionXY] = None, 
                          in_sets_validation: List[DataCollectionXY] = None):

        if self.do_mp:  
            if in_sets_training is None:
                in_sets_training = list()
            if in_sets_validation is None:
                in_sets_validation = list()

            SharedMemoryManager.save(in_observations, self.prefix + self.name + "_observations.arrow")

            idx_set = 0
            for set_training, set_validation in zip(in_sets_training, in_sets_validation):
                SharedMemoryManager.save(set_training, self.prefix + self.name + "_training_" + str(idx_set) + ".arrow")
                SharedMemoryManager.save(set_validation, self.prefix + self.name + "_validation_" + str(idx_set) + ".arrow")
                idx_set += 1

            self.n_sets = idx_set

        else:
            self.observations = in_observations
            self.sets_training = in_sets_training
            self.sets_validation = in_sets_validation

    def load_observations(self):

        if self.do_mp:
            sets_training = list()
            sets_validation = list()

            observations = SharedMemoryManager.load(self.prefix + self.name + "_observations.arrow")
            for idx_set in range(self.n_sets):
                set_training = SharedMemoryManager.load(self.prefix + self.name + "_training_" + str(idx_set) + ".arrow")
                set_validation = SharedMemoryManager.load(self.prefix + self.name + "_validation_" + str(idx_set) + ".arrow")
                sets_training.append(set_training)
                sets_validation.append(set_validation)

        else:
            observations = self.observations
            sets_training = self.sets_training
            sets_validation = self.sets_validation

        return observations, sets_training, sets_validation

    def del_observations(self):
        if self.do_mp:
            filepath = self.prefix + self.name + "_observations.arrow"
            os.remove(filepath)
            for idx_set in range(self.n_sets):
                filepath = self.prefix + self.name + "_training_" + str(idx_set) + ".arrow"
                os.remove(filepath)
                filepath = self.prefix + self.name + "_validation_" + str(idx_set) + ".arrow"
                os.remove(filepath)

    def decrement_uses(self):
        self.uses -= 1
        if self.uses <= 0:
            self.del_observations()





# TODO: Consider how the data is best stored, including preallocated arrays.
# TODO: Update comments. Update attributes.
class DataStorage:
    """
    A container that manages collections of data used for machine learning processes.
    """
    
    def __init__(self):
        log.info("%s - Initialising DataStorage." % Timestamp())

        # Store observations/queries as dicts of data collections keyed by collection IDs.
        # These collection IDs map to unique combinations of user-specified tags.
        # Each tag is a pair of tag key and tag value.
        # For example... 1 -> {"a":"1"} and 2 -> {"a":"1", "b":"2"}
        # In reverse, a map is kept of individual tags to associated collection IDs.
        # For example... {"a":"1"} -> [1, 2] and {"b":"2"} -> [2]
        # If no tags are provided, data is keyed to a no-tag ID.
        self.observations = dict()
        self.observations_tag_to_collection_ids = dict()    # Structure: dict[key_tag][value_tag] = list()
        self.observations_collection_id_to_tag_combos = dict()
        self.queries = dict()
        self.queries_tag_to_collection_ids = dict()         # Structure: dict[key_tag][value_tag] = list()
        self.queries_collection_id_to_tag_combos = dict()
        self.collection_id_no_tag = 0
        self.collection_id_new = 1

        # Keep track of the unique id that the last instance of data was given.
        self.id_data_last = -1
        
        # # Ingested data arrives from data ports.
        # # For data port X, this data is sent as a list of elements.
        # # The elements have keys: X_0, X_1, etc.
        # # Define a dict that links port-specific keys to storage-specific keys.
        # # This directs elements of incoming data to the right list.
        # self.ikeys_to_dkeys = dict()
        
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

    def get_keys(self):
        """
        Iterate through all collections of observations and return a set of data keys.
        """
        keys = set()
        for collection in self.observations.values():
            keys = keys | set(collection.data.column_names)

        return keys
    
    def get_tag_values(self, in_key_tag: str, as_query: bool = False):
        """
        Return a list of all possible values a data-storage tag has.
        """
        if not as_query:
            tag_to_collection_ids = self.observations_tag_to_collection_ids
        else:
            tag_to_collection_ids = self.queries_tag_to_collection_ids

        return list(tag_to_collection_ids[in_key_tag].keys())
    
    def get_tag_combo_from_collection_id(self, in_collection_id: int, 
                                         as_string: bool = True, as_query: bool = False):

        if in_collection_id == self.collection_id_no_tag:
            if as_string:
                return ""
            else:
                return None
        else:
            if not as_query:
                tag_combo = self.observations_collection_id_to_tag_combos[in_collection_id]
            else:
                tag_combo = self.queries_collection_id_to_tag_combos[in_collection_id]
            if as_string:
                tag_combo = "&".join(["(%s==%s)" % (key_tag, value_tag) for key_tag, value_tag in tag_combo.items()])
            return tag_combo
    
    def get_collection_ids(self, in_tags: Dict[str, str] = None, 
                           do_prepare_dicts: bool = False, do_exact_tag_combo: bool = False,
                           as_query: bool = False):
        """
        Returns a set of collection IDs associated with a dictionary of tags.
        If the 'exact tag combo' option is enabled, only one ID for exact tags is returned.
        For example, assume collection 1 is for {"a":"1"} and 2 is for {"a":"1", "b":"2"}.
        Requiring an exact tag combo while giving tag {"a":"1"} will only return collection 1, not 1 and 2.
        The 'prepare dicts' option assumes a DataCollection should exist for every tag combo.
        Therefore, it will create one if it does not exist.
        """

        if not as_query:
            source = self.observations
            collection_id_to_tag_combos = self.observations_collection_id_to_tag_combos
            tag_to_collection_ids = self.observations_tag_to_collection_ids
        else:
            source = self.queries
            collection_id_to_tag_combos = self.queries_collection_id_to_tag_combos
            tag_to_collection_ids = self.queries_tag_to_collection_ids
    
        # If no tags are supplied, the ID should refer to the standard collection.
        if in_tags is None or len(in_tags) == 0:
            set_collection_ids = set([self.collection_id_no_tag])

            if do_prepare_dicts:
                if not self.collection_id_no_tag in source:
                    source[self.collection_id_no_tag] = DataCollection()

            return set_collection_ids
        
        # Find all collections that are associated with the combination of tags.
        set_collection_ids = None
        for key_tag, value_tag in in_tags.items():

            if do_prepare_dicts:
                if not key_tag in tag_to_collection_ids:
                    tag_to_collection_ids[key_tag] = dict()
                if not value_tag in tag_to_collection_ids[key_tag]:
                    tag_to_collection_ids[key_tag][value_tag] = set()

            if set_collection_ids is None:
                set_collection_ids = tag_to_collection_ids[key_tag][value_tag]
            else:
                set_collection_ids = set_collection_ids & tag_to_collection_ids[key_tag][value_tag]

        if do_exact_tag_combo:
             # Go through the collections associated with the tags and find the one with exact tag combo.
            set_collection_ids_old = set_collection_ids
            set_collection_ids = set()
            for collection_id in set_collection_ids_old:
                if collection_id_to_tag_combos[collection_id] == in_tags:
                    set_collection_ids = set([collection_id])
                    break

        if do_prepare_dicts:
            # If no collection associates with the tag combo, optionally create one.
            if len(set_collection_ids) == 0:
                source[self.collection_id_new] = DataCollection()
                for key_tag, value_tag in in_tags.items():
                    tag_to_collection_ids[key_tag][value_tag].add(self.collection_id_new)
                collection_id_to_tag_combos[self.collection_id_new] = in_tags
                set_collection_ids = set([self.collection_id_new])
                self.collection_id_new += 1

        return set_collection_ids
    
    def get_amount(self, from_queries: bool = False):
        """
        Get the total amount of observations or queries that currently exist in storage.
        """
        if not from_queries:
            source = self.observations
        else:
            source = self.queries

        amount = 0
        for collection in source.values():
            amount += collection.get_amount()
        return amount
    
    # def get_collection(self, in_tags_inclusive = None, in_tags_exclusive = None):
    #     """
    #     Returns a concatenation of data collections linked/unlinked with 'inclusive/exclusive' tags.
    #     If there are no inclusive tags, all collections are selected prior to exclusions.
    #     """
    #     # if in_tags_inclusive is None:
    #     #     in_tags_inclusive = dict()
    #     # if in_tags_exclusive is None:
    #     #     in_tags_exclusive = dict()
        
    #     # if len(in_tags_inclusive) == 0:
    #     #     set_collection_ids = set(self.observations.keys())
    #     # else:
    #     #     set_collection_ids = set()

    #     # for key_tag, value_tag in in_tags_inclusive.items():
    #     #     set_collection_ids = set_collection_ids | self.tag_to_collection_ids[key_tag][value_tag]
    #     # for key_tag, value_tag in in_tags_exclusive.items():
    #     #     set_collection_ids = set_collection_ids - self.tag_to_collection_ids[key_tag][value_tag]

    #     # collection = None
    #     # for collection_id in list(set_collection_ids):
    #     #     if collection is None:
    #     #         collection = self.observations[collection_id]
    #     #     else:
    #     #         collection = collection + self.observations[collection_id]

    #     # return collection
    #     return get_collection(self.observations, self.tag_to_collection_ids, in_tags_inclusive, in_tags_exclusive)
    
    # def expand_collections(self, in_key):
    #     """
    #     Add a new keyed list of None values to every collection in storage with appropriate size.
    #     """
    #     for source in [self.observations, self.queries]:
    #         for collection in source.values():
    #             collection.data[in_key] = [None]*len(collection.timestamps)
    #     self.data_types[in_key] = None

    def store_data(self, in_data: pa.Table, in_tags: Dict[str, str], as_query: bool = False):

        # Convert user-specified tags to a unique collection ID.
        set_collection_ids = self.get_collection_ids(in_tags = in_tags, 
                                                     do_prepare_dicts = True, 
                                                     do_exact_tag_combo = True,
                                                     as_query = as_query)
        collection_id = list(set_collection_ids)[0]

        # Insert the data into the appropriate collection.
        if as_query:
            data_collection = self.queries[collection_id]
        else:
            data_collection = self.observations[collection_id]
        self.id_data_last = data_collection.insert(in_data, self.id_data_last)

        # Flick a switch so that learners can start ingesting new data.
        # Note: Resolving awaited futures are priority microtasks.
        # The following reset runs after the learners are signalled.
        if as_query:
            # self.queries = pa.concat_tables([self.queries, in_data], promote=True)
            self.has_new_queries.set_result(True)
            self.has_new_queries = asyncio.Future()
        else:
            # self.observations = pa.concat_tables([self.observations, in_data], promote=True)
            self.has_new_observations.set_result(True)
            self.has_new_observations = asyncio.Future()

    # # TODO: Update info logging once terminology is settled. Fix logging for tags.
    # def store_data(self, in_timestamp: Timestamp, in_data_port_id, in_keys, in_elements, in_data_types, 
    #                in_tags: Dict[str, str],
    #                as_query: bool = False):

    #     # Convert user-specified tags to a unique collection ID.
    #     set_collection_ids = self.get_collection_ids(in_tags = in_tags, 
    #                                                  do_prepare_dicts = True, 
    #                                                  do_exact_tag_combo = True)
    #     collection_id = list(set_collection_ids)[0]

    #     if as_query:
    #         timestamps = self.queries[collection_id].timestamps
    #         dict_storage = self.queries[collection_id].data
    #     else:
    #         timestamps = self.observations[collection_id].timestamps
    #         dict_storage = self.observations[collection_id].data
            
    #     timestamps.append(in_timestamp)

    #     # Extend all existing data or query lists by one empty slot.
    #     for dkey in self.get_key_dict():
    #         dict_storage[dkey].append(None)
        
    #     count_ikey_new = 0
    #     count_dkey_new = 0
    #     for key, element, data_type in zip(in_keys, in_elements, in_data_types):
            
    #         ikey = in_data_port_id + "_" + key
            
    #         # If a new port-specific key is encountered, initialise a list.
    #         # The list is initially named identically to this key.
    #         if not ikey in self.ikeys_to_dkeys:
    #             if count_ikey_new < SS.MAX_ALERTS_IKEY_NEW:
    #                 log.info("%s   DataStorage is newly encountering elements "
    #                          "of data/queries from a DataPort with key '%s'."
    #                          % (Timestamp(None), ikey))
    #             # TODO: Set up a safety mode where key is a distinct ikey.
    #             self.ikeys_to_dkeys[ikey] = key
    #             count_ikey_new += 1
            
    #         dkey = self.ikeys_to_dkeys[ikey]

    #         if not dkey in self.get_key_dict():
    #             if count_dkey_new < SS.MAX_ALERTS_DKEY_NEW:
    #                 log.info("%s   DataStorage has begun storing elements of "
    #                          "data/queries in a list with key '%s'." 
    #                          % (Timestamp(None), dkey))
    #             self.expand_collections(in_key = dkey)
                
    #             # If the type has not been determined elsewhere, the first element decides.
    #             # TODO: Improve inference process. Maybe update as new data is encountered.
    #             if data_type is None:
    #                 self.data_types[dkey] = infer_data_type(element)
    #             else:
    #                 self.data_types[dkey] = data_type
    #             count_dkey_new += 1
            
    #         # # Both data/queries must have the same keys.
    #         # if not dkey in self.observations[collection_id].data:
    #         #     if count_dkey_new < SS.MAX_ALERTS_DKEY_NEW:
    #         #         log.info("%s   DataStorage has begun storing elements of "
    #         #                  "data/queries in a list with key '%s'." 
    #         #                  % (Timestamp(None), dkey))
    #         #     self.observations[collection_id].data[dkey] = [None]*len(self.observations[collection_id].timestamps)
    #         #     self.queries[collection_id].data[dkey] = [None]*len(self.queries[collection_id].timestamps)
                
    #         #     # If the type has not been determined elsewhere, the first element decides.
    #         #     # TODO: Improve inference process. Maybe update as new data is encountered.
    #         #     if data_type is None:
    #         #         self.data_types[dkey] = infer_data_type(element)
    #         #     else:
    #         #         self.data_types[dkey] = data_type
    #         #     count_dkey_new += 1
            
    #         # Add the new element to the list with str-to-type conversion.
    #         try:
    #             # TODO: Some data types do not convert, e.g. NoneType. Consider how to fix/avoid.
    #             dict_storage[dkey][-1] = self.data_types[dkey].convert(element)
    #         except Exception as e:
    #             # TODO: Handle changes in data type for messy datasets.
    #             raise e
                
    #     if count_ikey_new > SS.MAX_ALERTS_IKEY_NEW:
    #         log.info("%s   In total, DataStorage has newly encountered data/queries "
    #                  "from a DataPort with %i unseen keys."
    #                  % (Timestamp(None), count_ikey_new))
    #     if count_dkey_new > SS.MAX_ALERTS_DKEY_NEW:
    #         log.info("%s   In total, DataStorage has begun storing data/queries "
    #                  "in %i new keyed lists."
    #                  % (Timestamp(None), count_dkey_new))
        
    #     # Flick a switch so that learners can start ingesting new data.
    #     # Note: Resolving awaited futures are priority microtasks.
    #     # The following reset runs after the learners are signalled.
    #     if as_query:
    #         self.has_new_queries.set_result(in_tags)
    #         self.has_new_queries = asyncio.Future()
    #     else:
    #         self.has_new_observations.set_result(in_tags)
    #         self.has_new_observations = asyncio.Future()


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
        Utility method to give user info about storage.
        """
        log.info("Stored data is arranged as follows.\n")
        for collection_id in self.observations:
            tag_combo =  self.get_tag_combo_from_collection_id(collection_id)
            if tag_combo == "":
                tag_text = ""
            else:
                tag_text = ", tag combo '%s'" % tag_combo
            log.info("- Observation collection %i%s.\n" % (collection_id, tag_text))
            log.info(self.observations[collection_id].data)
        for collection_id in self.queries:
            tag_combo =  self.get_tag_combo_from_collection_id(collection_id, as_query = True)
            if tag_combo == "":
                tag_text = ""
            else:
                tag_text = ", tag combo '%s'" % tag_combo
            log.info("- Query collection %i%s.\n" % (collection_id, tag_text))
            log.info(self.queries[collection_id].data)



    # def info(self):
    #     """
    #     Utility method to give user info about data ports and storage.
    #     """
    #     if len(self.data_types.keys()) > SS.MAX_INFO_KEYS_EXAMPLE:
    #         data_types_keys = list(self.data_types.keys())
    #         len_start = int(np.ceil(SS.MAX_INFO_KEYS_EXAMPLE/2))
    #         len_end = SS.MAX_INFO_KEYS_EXAMPLE - len_start
    #         example_keys = ", ".join(key + " (" + self.data_types[key].to_string() + ")"
    #                                  for key in data_types_keys[:len_start])
    #         example_keys += ", ..., "
    #         example_keys += ", ".join(key + " (" + self.data_types[key].to_string() + ")"
    #                                   for key in data_types_keys[-len_end:])
    #     else:
    #         example_keys = ", ".join(key + " (" + self.data_types[key].to_string() + ")"
    #                                  for key in self.data_types)
            
    #     if len(self.ikeys_to_dkeys) > SS.MAX_INFO_PIPE_EXAMPLE:
    #         ikeys_to_dkeys_keys = list(self.ikeys_to_dkeys.keys())
    #         len_start = int(np.ceil(SS.MAX_INFO_PIPE_EXAMPLE/2))
    #         len_end = SS.MAX_INFO_PIPE_EXAMPLE - len_start
    #         example_pipe = ", ".join("{" + key + " -> " + self.ikeys_to_dkeys[key] + "}"
    #                                  for key in ikeys_to_dkeys_keys[:len_start])
    #         example_pipe += ", ..., "
    #         example_pipe += ", ".join("{" + key + " -> " + self.ikeys_to_dkeys[key] + "}"
    #                                   for key in ikeys_to_dkeys_keys[-len_end:])
    #     else:
    #         example_pipe = ", ".join("{" + key + " -> " + self.ikeys_to_dkeys[key] + "}"
    #                                  for key in self.ikeys_to_dkeys)

        
    #     log.info("Stored data is arranged into lists identified as follows.")
    #     log.info("Keys: %s" % example_keys)
    #     log.info("DataPorts pipe data into the lists as follows.")
    #     log.info("Pipe: %s" % example_pipe)
        
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