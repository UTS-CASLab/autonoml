# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 19:20:56 2023

@author: David J. Kedziora
"""

from enum import Enum

from sklearn import metrics

class LossFunction(Enum):
    RMSE = 0

def calculate_loss(y_response, y_true, in_loss_function: LossFunction = None):
        """
        Compare a set of predictions against a set of expected values.
        The output must be a loss.
        """
        if in_loss_function == LossFunction.RMSE:
            val = metrics.mean_squared_error(y_true = y_true, y_pred = y_response, 
                                             squared = False)
        else:
            raise NotImplementedError
        
        return val