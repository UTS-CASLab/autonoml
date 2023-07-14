# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 20:00:06 2023

@author: David J. Kedziora
"""

from .settings import SystemSettings as SS

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_feature_importance(in_keys_features, in_importance):

    x_max = len(in_keys_features)
    
    fig, ax = plt.subplots()
    
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
    
    analysis = pd.DataFrame(data={
        "Model Response": in_vals_response,
        "True Values": in_vals_true
    })
    
    val_min = min(min(in_vals_response), min(in_vals_true))
    val_max = max(max(in_vals_response), max(in_vals_true))
    
    g = sns.JointGrid(data = analysis, x = "Model Response", y = "True Values")
    # bins = np.linspace(val_min, val_max, SS.BINS_HIST + 1)
    g.plot_joint(sns.histplot, cmap = "plasma",
                 bins = SS.BINS_HIST, binrange = (val_min, val_max))
    g.plot_marginals(sns.histplot, kde = True,
                     bins = SS.BINS_HIST, binrange = (val_min, val_max))
    # Note: Use color steelblue to more closely match default marginal plots.
    g.ax_joint.plot([val_min, val_max], [val_min, val_max], 
                    color = "black", linestyle = "dashed")
    g.ax_joint.set_xlim([val_min, val_max])
    g.ax_joint.set_ylim([val_min, val_max])
    
    # g.ax_joint.
    
    # fig, ax = plt.subplots()
    # ax.hist2d(in_vals_response, in_vals_true, bins = SS.BINS_HIST, 
    #           cmin = 1, cmap = "plasma")
    # ax.plot([val_min, val_max], [val_min, val_max], "k:")
    # if in_title:
    #     ax.set_title(in_title)
    # ax.axis("equal")
    # ax.set_xlim([val_min, val_max])
    # ax.set_ylim([val_min, val_max])
    # ax.set_xlabel("Model Response")
    # ax.set_ylabel("True Values")
    # plt.show()