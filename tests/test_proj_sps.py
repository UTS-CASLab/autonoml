# -*- coding: utf-8 -*-
"""
This script tests AutonoML run on data that arrives in three different ways.
It involves two SPS datasets of 500 instances each, referred to here as A and B.
The three modes are...
- Batch: Ingest all of A as a batch. Then ingest all of B as a batch.
- Hybrid: Ingest all of A as a batch. Then stream all of B.
- Stream: Stream all of A. Then stream all of B.

Created on Sun Nov 26 18:41:29 2023

@author: David J. Kedziora
"""
import autonoml as aml
import subprocess

import csv
import time

if __name__ == '__main__':

    # Define hostname/ports for three data streamers.
    hostname = aml.SystemSettings.DEFAULT_HOSTNAME
    port_stream_A = aml.SystemSettings.DEFAULT_PORT_OBSERVATIONS
    port_stream_B = aml.SystemSettings.DEFAULT_PORT_OBSERVATIONS + 10
    port_hybrid_B = aml.SystemSettings.DEFAULT_PORT_OBSERVATIONS + 20

    # Define the data files.
    filename_A = "./data/sps/test_sps_quality_1000_events_1p2uW_3000cps.csv"
    filename_B = "./data/sps/test_sps_quality_1000_events_30uW_7000cps.csv"

    # Name the data/log file for the streamer subprocess as well as the broadcasting period/delay.
    filename_log_stream_A = "./test_proj_sps_stream_A.log"
    filename_log_stream_B = "./test_proj_sps_stream_B.log"
    filename_log_hybrid_B = "./test_proj_sps_hybrid_B.log"
    period_data_stream = 1.0
    delay_before_start = 1.0

    # As AutonoML operations choke up, client confirmations to broadcasters choke up.
    # This is a manual fix, although confirmations may eventually operate on their own thread/process.
    delay_for_server_abandon = 150

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

    # Ingest data directly from file as required.
    proj.ingest_file(filename_A, in_tags = {"mode":"batch"})
    proj.ingest_file(filename_B, in_tags = {"mode":"batch"})
    proj.ingest_file(filename_A, in_tags = {"mode":"hybrid"})

    # Connect the AutonoMachine to the expected data-broadcasting servers.
    # Because the user provides preset field names, the storage will begin upon connection.
    # Note: The ports should keep trying to reconnect unless deleted.
    port_1 = proj.ingest_stream(hostname, port_stream_A, 
                                in_id_stream = "stream_A", 
                                in_field_names = get_field_names(filename_A),
                                in_tags = {"mode":"stream"})
    port_2 = proj.ingest_stream(hostname, port_stream_B, 
                                in_id_stream = "stream_B", 
                                in_field_names = get_field_names(filename_B),
                                in_tags = {"mode":"stream"})
    port_3 = proj.ingest_stream(hostname, port_hybrid_B, 
                                in_id_stream = "hybrid_B", 
                                in_field_names = get_field_names(filename_B),
                                in_tags = {"mode":"hybrid"})

    # Simulate the streams by running separate processes for the data broadcaster.
    with open(filename_log_stream_A, "w") as file_log_streamer:
        server_process = subprocess.Popen(["python", "sim_stream.py", 
                                           "--filepath_data", filename_A, 
                                           "--period_data_stream", str(period_data_stream),
                                           "--delay_before_start", str(delay_before_start),
                                           "--hostname_observations", hostname,
                                           "--port_observations", str(port_stream_A),
                                           "--delay_for_server_abandon", str(delay_for_server_abandon)],
                                          stdout = file_log_streamer, 
                                          stderr = subprocess.STDOUT,
                                          universal_newlines = True)
    with open(filename_log_stream_B, "w") as file_log_streamer:
        server_process = subprocess.Popen(["python", "sim_stream.py", 
                                           "--filepath_data", filename_B, 
                                           "--period_data_stream", str(period_data_stream),
                                           "--delay_before_start", str(delay_before_start),
                                           "--hostname_observations", hostname,
                                           "--port_observations", str(port_stream_B),
                                           "--delay_for_server_abandon", str(delay_for_server_abandon)],
                                          stdout = file_log_streamer, 
                                          stderr = subprocess.STDOUT,
                                          universal_newlines = True)
    with open(filename_log_hybrid_B, "w") as file_log_streamer:
        server_process = subprocess.Popen(["python", "sim_stream.py", 
                                           "--filepath_data", filename_B, 
                                           "--period_data_stream", str(period_data_stream),
                                           "--delay_before_start", str(delay_before_start),
                                           "--hostname_observations", hostname,
                                           "--port_observations", str(port_hybrid_B),
                                           "--delay_for_server_abandon", str(delay_for_server_abandon)],
                                          stdout = file_log_streamer, 
                                          stderr = subprocess.STDOUT,
                                          universal_newlines = True)
    
    # Wait for the ports to be connected before proceeding.
    while not (port_1.is_connected() and port_2.is_connected() and port_3.is_connected()):
        time.sleep(1)
    time.sleep(1)   # Wait a moment for the first observations to be stored.

    # # Sanity-check that the initial data looks fine in storage.
    # proj.info_storage()

    # Import the appropriate strategy file.
    strategy = aml.import_strategy("./test_proj.strat")

    # Start learning the target variable.
    proj.learn(in_key_target = "estimate",
               in_keys_features = ["best"], do_exclude = True,
               in_strategy = strategy,
               in_tags_allocation = ["mode"])