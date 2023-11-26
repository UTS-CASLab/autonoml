# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:12:44 2023

@author: David J. Kedziora
"""

import autonoml as aml

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Simulate data streaming.")
    parser.add_argument("--filename_data", type = str, default="./data/dummy/train_dummy.csv",
                        help = "Path to the data file.")
    parser.add_argument("--period_data_stream", type = float, default = 1.0,
                        help = "Time period for data streaming in seconds.")
    parser.add_argument("--delay_before_start", type = float, default = 0.0,
                        help = "Time delay before broadcasting starts in seconds.")
    return parser.parse_args()

filename_data = "./data/pharma/indpensim_batch.csv"

if __name__ == "__main__":
    args = parse_arguments()

    streamer = aml.SimDataStreamer(args.filename_data, 
                                   in_period_data_stream = args.period_data_stream,
                                   in_delay_before_start = args.delay_before_start)