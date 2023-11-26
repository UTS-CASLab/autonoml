# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:41:15 2023

@author: David J. Kedziora
"""

import autonoml as aml

filename_data = "./data/drift/mixed_0101_abrupto.csv"

# Data file interleaves class representation at 1:1 ratio.
# So, choose a training/testing ratio of 4:1.
streamer = aml.SimDataStreamer(filename_data, in_observations_per_query = 4)