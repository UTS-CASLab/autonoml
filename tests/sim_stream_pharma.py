# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:12:44 2023

@author: David J. Kedziora
"""

import autonoml as aml

filename_data = "./data/pharma/indpensim_batch.csv"

streamer = aml.SimDataStreamer(filename_data, in_period_data_stream = 1.0,
                               in_delay_before_start = 1.0)