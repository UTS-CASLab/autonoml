# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:58:33 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp, asyncio_task_from_method
from .pool import (OnlineLinearRegressor, 
                   PartialLeastSquaresRegressor)

import asyncio

class TaskSolver:
    """
    A wrapper for components that learn from data and respond to queries.
    """
    
    def __init__(self, in_data_storage, in_key_target, 
                 in_keys_features = None, do_exclude = False):
        log.info("%s - A TaskSolver has been initialised." % Timestamp())
        
        self.data_storage = in_data_storage
        
        o1, o2 = self.set_target_and_features(in_key_target = in_key_target,
                                              in_keys_features = in_keys_features,
                                              do_exclude = do_exclude)
        self.key_target = o1
        self.keys_features = o2
        
        self.pipelines = list()
        
        # Keep track of the data-storage instance up to which model has used.
        # TODO: Consider variant starting points for the model and update log messages.
        self.count_data = 0
        self.count_queries = 0
        
        # Set up a variable that can be awaited elsewhere.
        # This 'switch', when flicked, signals that the pipelines can be queried.
        self.can_query = asyncio.Future()
        
        self.ops = None
        asyncio_task_from_method(self.gather_ops)
        
    async def gather_ops(self):
        self.ops = [asyncio_task_from_method(op) for op in [self.process_strategy,
                                                            self.process_queries]]
        await asyncio.gather(*self.ops)
        #, return_exceptions=True)
        
    def stop(self):
        # Cancel all asynchronous operations.
        if self.ops:
            for op in self.ops:
                op.cancel()
        
    async def process_strategy(self):
        self.pipelines.extend([OnlineLinearRegressor(),
                               PartialLeastSquaresRegressor()])
        
        # self.pipelines.append(PartialLeastSquaresRegressor())
        
        self.can_query.set_result(True)
        
        while True:
            # Check for new data and learn from it.
            if self.count_data < len(self.data_storage.timestamps_data):
                self.count_data = len(self.data_storage.timestamps_data)
                
                # df = self.data_storage.get_dataframe()
                # print(df)
                # df = df.sample(frac = 1)
                # print(df)
                        
                for pipeline in self.pipelines:
                    # TODO: Develop for an actual pipeline.
                    component = pipeline
                    time_start = Timestamp().time
                    x, y = self.data_storage.get_data(in_keys_features = self.keys_features,
                                                      in_key_target = self.key_target,
                                                      in_format = component.data_format)
                    component.learn(x, y)
                    score = component.score(x, y, do_remember = True, for_training = True)
                    time_end = Timestamp().time
                    y_last = y[-1:]
                    y_pred_last = component.query(x[-1:])
            
                    log.info("%s - Model '%s' has learned from a total of %i observations.\n"
                             "%s   Time taken to retrieve, learn and score model on data: %.3f s\n"
                             "%s   Score on those observations: %f\n"
                             "%s   Last observation: Prediction '%s' vs True Value '%s'"
                             % (Timestamp(), component.name, self.count_data,
                                Timestamp(None), time_end - time_start,
                                Timestamp(None), score,
                                Timestamp(None), y_pred_last, y_last))
                
            await self.data_storage.has_new_data
            
    async def process_queries(self):
        
        while True:
            await self.can_query
            
            # Check for new queries and derive responses.
            # Score them if possible.
            if self.count_queries < len(self.data_storage.timestamps_queries):
                self.count_queries = len(self.data_storage.timestamps_queries)
                
                # df = self.data_storage.get_dataframe()
                # print(df)
                # df = df.sample(frac = 1)
                # print(df)
                        
                for pipeline in self.pipelines:
                    # TODO: Develop for an actual pipeline.
                    component = pipeline
                    time_start = Timestamp().time
                    x, y = self.data_storage.get_data(in_keys_features = self.keys_features,
                                                      in_key_target = self.key_target,
                                                      in_format = component.data_format,
                                                      from_queries = True)
                    score = component.score(x, y, do_remember = True)
                    time_end = Timestamp().time
                    y_last = y[-1:]
                    y_pred_last = component.query(x[-1:])
            
                    log.info("%s - Model '%s' has responded to a total of %i queries.\n"
                             "%s   Time taken to retrieve and score model on queries: %.3f s\n"
                             "%s   Score on those queries: %f\n"
                             "%s   Last query: Prediction '%s' vs True Value '%s'"
                             % (Timestamp(), component.name, self.count_queries,
                                Timestamp(None), time_end - time_start,
                                Timestamp(None), score,
                                Timestamp(None), y_pred_last, y_last))
            
            await self.data_storage.has_new_queries
            
        # x = list()
        # y = self.data_storage.queries[self.key_target]
        # # TODO: Control DataStorage updates.
        # for key in self.keys_features:
        #     x_element = self.data_storage.queries[key]
        #     x.append(x_element)
        # x = [list(row) for row in zip(*x)]  # Transpose.
        
        # for pipeline in self.pipelines:
        #     # TODO: Develop for an actual pipeline.
        #     component = pipeline
        #     score = component.score(x, y)
        #     y_last = y[-1]
        #     y_pred_last = component.query([x[-1]])[0][0]
            
        #     log.info("%s - Model '%s' has been tested on a total of %i queries with expected responses." 
        #              % (Timestamp(), component.name, len(y)))
        #     log.info("%s   Score on those testable queries: %f" 
        #              % (Timestamp(None), score))
        #     log.info("%s   Last query: Prediction '%s' vs True Value '%s'" 
        #              % (Timestamp(None), y_pred_last, y_last))
            
    # TODO: Include error checking for no features. Error-check target existence somewhere too.
    def set_target_and_features(self, in_key_target, 
                                in_keys_features = None, do_exclude = False):
        
        if in_key_target in self.data_storage.data:
            key_target = in_key_target
        else:
            text_error = "Desired target key '%s' cannot be found in DataStorage." % in_key_target 
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
        
        keys_features = list()
        # If the user provided feature keys, but not to exclude...
        # Include them as long as such features exist in the data storage.
        if in_keys_features and not do_exclude:
            for key_feature in in_keys_features:
                if key_feature in self.data_storage.data:
                    keys_features.append(key_feature)
                else:
                    log.warning("%s - Desired feature key '%s' cannot be found in DataStorage.\n"
                                "%s   The TaskSolver will ignore it." 
                                % (Timestamp(), key_feature, Timestamp(None)))
        # Otherwise, include every feature existing in the data storage...
        # Except for feature keys specified with the intention of excluding.
        else:
            for dkey in self.data_storage.data.keys():
                if not dkey == in_key_target:
                    if do_exclude and dkey in in_keys_features:
                        log.info("%s - DataStorage key '%s' has been marked as not a feature.\n"
                                 "%s   The TaskSolver will ignore it."
                                 % (Timestamp(), dkey, Timestamp(None)))
                    else:
                        keys_features.append(dkey)
        
        return key_target, keys_features


    def info(self):
        """
        Utility method to give user info about the task solver and its models.
        """
        
        for pipeline in self.pipelines:
            # TODO: Develop for an actual pipeline.
            component = pipeline
            component.inspect_structure(self.keys_features)
            component.inspect_performance(for_training = True)
            component.inspect_performance(for_training = False)
        

# class TaskSolver:
#     """
#     A wrapper for components that learn from data and respond to queries.
#     """
    
#     def __init__(self, in_key_target, in_data_storage):
#         log.info("%s - A TaskSolver has been initialised." % Timestamp())
        
#         self.data_storage = in_data_storage
        
#         self.key_target = in_key_target
        
#         self.model = linear_model.LogisticRegression()
#         self.metric = metrics.RMSE()
        
#         # Keep track of the data-storage instance up to which model has used.
#         # TODO: Consider variant starting points for the model and update log messages.
#         self.count_data = 0
        
#         self.task = asyncio.get_event_loop().create_task(self.process_strategy())
        
#     def stop(self):
#         self.task.cancel()
        
#     async def process_strategy(self):
#         while True:
#             # Check for new data and learn from it.
#             count_instance = 0
#             y = None
#             y_pred = None
#             while self.count_data < len(self.data_storage.timestamps):
#                 x = dict()
#                 for key in self.data_storage.data:
#                     if key == self.key_target:
#                         y = self.data_storage.data[key][self.count_data]
#                     else:
#                         x[key] = self.data_storage.data[key][self.count_data]
                        
#                 y_pred = self.model.predict_one(x)
#                 self.metric = self.metric.update(y, y_pred)
#                 self.model.learn_one(x, y)
                
#                 self.count_data += 1
#                 count_instance += 1
            
#             if count_instance > 0:
#                 log.info("%s - The TaskSolver has learned from another %i observations." 
#                          % (Timestamp(), count_instance))
#                 log.info("%s   Metric is %f after Observation %i"
#                          % (Timestamp(None), self.metric.get(), self.count_data))
#                 log.info("%s   Last observation: Prediction '%s' vs True Value '%s'" 
#                          % (Timestamp(None), y_pred, y))
                
#             await self.data_storage.has_new_data