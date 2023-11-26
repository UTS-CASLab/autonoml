# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:41:29 2023

@author: David J. Kedziora
"""
import autonoml as aml
import subprocess

import csv
import time

if __name__ == '__main__':

    # The data streamer will be broadcasting on the default host/port.
    # Define them here for the AutonoMachine to connect with.
    server_hostname = aml.SystemSettings.DEFAULT_HOSTNAME
    server_port_data = aml.SystemSettings.DEFAULT_PORT_DATA

    # Name the data/log file for the streamer subprocess as well as the broadcasting period.
    filename_data_streamer = "./data/sps/train_sps_quality_1000_events_1p2uW_3000cps.csv"
    filename_log_streamer = "./test_proj_sps_streamer.log"
    period_data_stream = 1.0
    delay_before_start = 1.0

    def get_field_names(in_filename_data):
        """
        When connecting to a data stream, knowledge of the metadata often comes from elsewhere.
        In this case, just grab the headers from the .csv file to be streamed.
        """
        with open(in_filename_data, "r") as file:
            csv_reader = csv.reader(file)
            field_names = next(csv_reader)

        return field_names

    # Start up the AutonoMachine.
    proj = aml.AutonoMachine()

    # Connect the AutonoMachine to the expected data-broadcasting server.
    # Because the user provides preset field names, the storage will begin upon connection.
    # Note: The port should keep trying to reconnect unless deleted.
    port = proj.ingest_stream(server_hostname, server_port_data, 
                              in_id_stream = "pharma", 
                              in_field_names = get_field_names(filename_data_streamer))

    # Simulate the stream by running a separate process for the data broadcaster.
    with open(filename_log_streamer, "w") as file_log_streamer:
        server_process = subprocess.Popen(["python", "sim_stream.py", 
                                           "--filename_data", filename_data_streamer, 
                                           "--period_data_stream", str(period_data_stream),
                                           "--delay_before_start", str(delay_before_start)],
                                          stdout = file_log_streamer, 
                                          stderr = subprocess.STDOUT,
                                          universal_newlines = True)
    
    # Wait for the port to be connected before proceeding.
    while not port.is_connected():
        time.sleep(1)
    time.sleep(1)   # Wait a moment for the first observations to be stored.

    # Sanity-check that the initial data looks fine in storage.
    # proj.info_storage()

    # Import the appropriate strategy file.
    strategy = aml.import_strategy("./test_proj_pharma.strat")

    # Start learning the target variable.
    proj.learn(in_key_target = "estimate",
               in_keys_features = ["best"], do_exclude = True,
               in_strategy = strategy)