# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:12:44 2023

@author: David J. Kedziora
"""

import autonoml as aml

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Simulate data streaming.")
    parser.add_argument("--filepath_data", type = str, default="./data/dummy/train_dummy.csv",
                        help = "Path to the data file.")
    parser.add_argument("--period_data_stream", type = float, default = 1.0,
                        help = "Time period for data streaming in seconds.")
    parser.add_argument("--delay_before_start", type = float, default = 0.0,
                        help = "Time delay before broadcasting starts in seconds.")
    parser.add_argument("--hostname_observations", default = aml.SystemSettings.DEFAULT_HOSTNAME,
                        help = "Hostname for the server broadcasting observations.")
    parser.add_argument("--port_observations", type = int, default = aml.SystemSettings.DEFAULT_PORT_OBSERVATIONS,
                        help = "Port number for the server broadcasting observations.")
    parser.add_argument("--delay_for_server_abandon", type = float, default = aml.SystemSettings.DELAY_FOR_SERVER_ABANDON,
                        help = "Time delay without client confirmation before broadcasting is abandoned.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    aml.SystemSettings.DELAY_FOR_SERVER_ABANDON = args.delay_for_server_abandon

    streamer = aml.SimDataStreamer(in_filepath_data = args.filepath_data, 
                                   in_period_data_stream = args.period_data_stream,
                                   in_delay_before_start = args.delay_before_start,
                                   in_hostname_observations = args.hostname_observations,
                                   in_port_observations = args.port_observations)