# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 20:00:06 2023

@author: David J. Kedziora
"""

from .settings import SystemSettings as SS

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_feature_importance(in_keys_features, in_importance):
    # print(in_keys_features)
    # print(in_importance)
    
    fig, ax = plt.subplots()
    
    # analysis = pd.DataFrame(data={
    #     "Feature": in_keys_features,
    #     "Importance": in_importance
    # })
    x_max = len(in_keys_features)
    
    ax.bar(x = in_keys_features, height = in_importance)
    ax.set_title("Feature Importance from Model Coefficients")
    if len(in_keys_features) > SS.MAX_LABELS_BAR:
        step = int(len(in_keys_features)/SS.MAX_LABELS_BAR + 1)
        ax.set_xticks(list(range(0, x_max, step)), in_keys_features[::step], rotation="vertical")
    else:
        ax.set_xticks(list(range(x_max)), in_keys_features, rotation="vertical")
    ax.set_xlim([0, x_max])
    plt.show()
    
def plot_performance(in_vals_response, in_vals_true, in_title = None):
    
    val_min = min(min(in_vals_response), min(in_vals_true))
    val_max = max(max(in_vals_response), max(in_vals_true))
    
    fig, ax = plt.subplots()
    ax.hist2d(in_vals_response, in_vals_true, bins = SS.BINS_HIST, 
              cmin = 1, cmap = "plasma")
    ax.plot([val_min, val_max], [val_min, val_max], "k:")
    if in_title:
        ax.set_title(in_title)
    ax.axis("equal")
    ax.set_xlim([val_min, val_max])
    ax.set_ylim([val_min, val_max])
    ax.set_xlabel("Model Response")
    ax.set_ylabel("True Values")
    plt.show()