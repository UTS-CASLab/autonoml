# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:09:18 2023

@author: David J. Kedziora
"""

import autonoml as aml
import subprocess

import csv
import time

# The data streamer will be broadcasting on the default host/port.
# Define them here for the AutonoMachine to connect with.
server_hostname = aml.SystemSettings.DEFAULT_HOSTNAME
server_port_data = aml.SystemSettings.DEFAULT_PORT_DATA
    
# Define a log file for the streamer subprocess.
filename_log_streamer = "test_proj_pharma_streamer.log"

def get_field_names():
    """
    When connecting to a data stream, knowledge of the metadata often comes from elsewhere.
    In this case, just grab the headers from the .csv file to be streamed.
    """
    filename_data = "./data/pharma/indpensim_batch.csv"

    with open(filename_data, 'r') as file:
        csv_reader = csv.reader(file)
        field_names = next(csv_reader)

    return field_names

if __name__ == '__main__':

    # Start up the AutonoMachine.
    proj = aml.AutonoMachine()

    # Connect the AutonoMachine to the expected data-broadcasting server.
    # Because the user provides preset field names, the storage will begin upon connection.
    # Note: The port should keep trying to reconnect unless deleted.
    port = proj.ingest_stream(server_hostname, server_port_data, 
                              in_id_stream = "pharma", in_field_names = get_field_names())

    # Simulate the stream by running a separate process for the data broadcaster.
    with open(filename_log_streamer, "w") as file_log_streamer:
        server_process = subprocess.Popen(["python", "sim_stream_pharma.py"],
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
    proj.learn(in_key_target = "Penicillin concentration(P:g/L)",
               in_keys_features = ["Time (h)"], do_exclude = True,
               in_strategy = strategy)