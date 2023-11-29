# -*- coding: utf-8 -*-
"""
Created on Fri May 12 18:35:47 2023

@author: David J. Kedziora
"""

import autonoml as aml

filename_data = "./data/drift/mixed_0101_gradual.csv"

port_observations = aml.SystemSettings.DEFAULT_PORT_OBSERVATIONS + 10
port_queries = aml.SystemSettings.DEFAULT_PORT_QUERIES + 10

# Data file interleaves class representation at 1:1 ratio.
# A training/testing ratio of 1.5:1 should be fine.
streamer = aml.SimDataStreamer(filename_data,
                               in_observations_per_query = 1.5,
                               in_period_data_stream = 0.75,
                               in_port_observations = port_observations,
                               in_port_queries = port_queries)