# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:09:18 2023

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
    server_port_observations = aml.SystemSettings.DEFAULT_PORT_OBSERVATIONS
    
    # Name the data/log file for the streamer subprocess as well as the broadcasting period.
    filepath_data_streamer = "./data/pharma/indpensim_batch.csv"
    filename_log_streamer = "./test_proj_pharma_streamer.log"
    period_data_stream = 1.0
    delay_before_start = 1.0

    def get_field_names(in_filepath_data):
        """
        When connecting to a data stream, knowledge of the metadata often comes from elsewhere.
        In this case, just grab the headers from the .csv file to be streamed.
        """
        with open(in_filepath_data, "r") as file:
            csv_reader = csv.reader(file)
            field_names = next(csv_reader)

        return field_names

    # Start up the AutonoMachine.
    proj = aml.AutonoMachine()

    # Connect the AutonoMachine to the expected data-broadcasting server.
    # Because the user provides preset field names, the storage will begin upon connection.
    # Note: The port should keep trying to reconnect unless deleted.
    port = proj.ingest_stream(server_hostname, server_port_observations, 
                              in_id_stream = "pharma", 
                              in_field_names = get_field_names(filepath_data_streamer))

    # Simulate the stream by running a separate process for the data broadcaster.
    with open(filename_log_streamer, "w") as file_log_streamer:
        server_process = subprocess.Popen(["python", "sim_stream.py", 
                                           "--filepath_data", filepath_data_streamer, 
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
    proj.info_storage()

    # Import the appropriate strategy file.
    strategy = aml.import_strategy("./test_proj_pharma.strat")

    # Start learning the target variable.
    proj.learn(in_key_target = "Penicillin concentration(P:g/L)",
               in_keys_features = ["Time (h)", 
                                   "PAA concentration offline(PAA_offline:PAA (g L^{-1}))",
                                   "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))",
                                   "Offline Penicillin concentration(P_offline:P(g L^{-1}))",
                                   "Offline Biomass concentratio(X_offline:X(g L^{-1}))",
                                   "Viscosity(Viscosity_offline:centPoise)"], 
               do_exclude = True,
               in_strategy = strategy)