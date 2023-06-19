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
        self.count_data = 0
        
        self.task = asyncio.get_event_loop().create_task(self.continuously_learn())
        
    def stop(self):
        self.task.cancel()
        
    async def continuously_learn(self):
        while True:
            # Check for new data and learn from it.
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
                
                log.info("%s - RMSE is %f after Observation %i"
                         " - True Value %s vs Prediction %s" 
                         % (Timestamp(), self.metric.get(), self.count_data,
                            y, y_pred))
                
                self.count_data += 1
                
            await self.data_storage.has_new_data