# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:58:33 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .pool import PartialLeastSquaresRegressor

import asyncio

class TaskSolver:
    """
    A wrapper for components that learn from data and respond to queries.
    """
    
    def __init__(self, in_key_target, in_data_storage):
        log.info("%s - A TaskSolver has been initialised." % Timestamp())
        
        self.data_storage = in_data_storage
        
        self.key_target = in_key_target
        
        self.pipelines = list()
        
        # Keep track of the data-storage instance up to which model has used.
        # TODO: Consider variant starting points for the model and update log messages.
        self.count_data = 0
        
        self.task = asyncio.get_event_loop().create_task(self.process_strategy())
        
    def stop(self):
        self.task.cancel()
        
    async def process_strategy(self):
        
        self.pipelines.append(PartialLeastSquaresRegressor())
        
        while True:
            # Check for new data and learn from it.
            if self.count_data < len(self.data_storage.timestamps):
                self.count_data = len(self.data_storage.timestamps)
                
                x = list()
                y = None
                # TODO: Make sure everything maintains order in all cases.
                for key in self.data_storage.data:
                    # TODO: Throw error if there is no key_target.
                    if key == self.key_target:
                        y = self.data_storage.data[key][:5]
                    else:
                        x_element = self.data_storage.data[key][:5]
                        x.append(x_element)
                x = [list(row) for row in zip(*x)]  # Transpose.
                    
                        
                for pipeline in self.pipelines:
                    # TODO: Develop for an actual pipeline.
                    component = pipeline
                    component.learn(x, y)
                    score = component.score(x, y)
                    y_last = y[-1]
                    y_pred_last = component.query([x[-1]])[0][0]
            
                    log.info("%s - Model '%s' has learned from a total of %i observations." 
                             % (Timestamp(), component.name, self.count_data))
                    log.info("%s   Score on those observations: %f" 
                             % (Timestamp(None), score))
                    log.info("%s   Last observation: Prediction '%s' vs True Value '%s'" 
                             % (Timestamp(None), y_pred_last, y_last))
                
            await self.data_storage.has_new_data

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