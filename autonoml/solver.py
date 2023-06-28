# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:58:33 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp

import asyncio

from river import linear_model
from river import metrics

class TaskSolver:
    """
    A wrapper for components that learn from data and respond to queries.
    """
    
    def __init__(self, in_key_target, in_data_storage):
        log.info("%s - A TaskSolver has been initialised." % Timestamp())
        
        self.data_storage = in_data_storage
        
        self.key_target = in_key_target
        
        self.model = linear_model.LogisticRegression()
        self.metric = metrics.RMSE()
        
        # Keep track of the data-storage instance up to which model has used.
        # TODO: Consider variant starting points for the model and update log messages.
        self.count_data = 0
        
        self.task = asyncio.get_event_loop().create_task(self.continuously_learn())
        
    def stop(self):
        self.task.cancel()
        
    async def continuously_learn(self):
        while True:
            # Check for new data and learn from it.
            count_instance = 0
            y = None
            y_pred = None
            while self.count_data < len(self.data_storage.timestamps):
                x = dict()
                for key in self.data_storage.data:
                    if key == self.key_target:
                        y = self.data_storage.data[key][self.count_data]
                    else:
                        x[key] = self.data_storage.data[key][self.count_data]
                        
                y_pred = self.model.predict_one(x)
                self.metric = self.metric.update(y, y_pred)
                self.model.learn_one(x, y)
                
                self.count_data += 1
                count_instance += 1
            
            if count_instance > 0:
                log.info("%s - The TaskSolver has learned from another %i observations." 
                         % (Timestamp(), count_instance))
                log.info("%s   Metric is %f after Observation %i"
                         % (Timestamp(None), self.metric.get(), self.count_data))
                log.info("%s   Last observation: Prediction %s vs True Value %s" 
                         % (Timestamp(None), y_pred, y))
                
            await self.data_storage.has_new_data