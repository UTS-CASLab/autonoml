# -*- coding: utf-8 -*-
"""
Created on Fri May 12 18:35:47 2023

@author: David J. Kedziora
"""

import autonoml as aml

filename_data = "./data/drift/mixed_0101_gradual.csv"

port_data = aml.SystemSettings.DEFAULT_PORT_DATA + 10
port_query = aml.SystemSettings.DEFAULT_PORT_QUERY + 10

# Data file interleaves class representation at 1:1 ratio.
# A training/testing ratio of 1.5:1 should be fine.
streamer = aml.SimDataStreamer(filename_data,
                               in_observations_per_query = 1.5,
                               in_period_data_stream = 0.75,
                               in_port_data = port_data,
                               in_port_query = port_query)