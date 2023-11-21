# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:09:18 2023

@author: David J. Kedziora
"""

import autonoml as aml
import subprocess

if __name__ == '__main__':
    
    # Define a default data server host/port for the user to connect with.
    server_hostname = aml.SystemSettings.DEFAULT_HOSTNAME
    server_port_data = aml.SystemSettings.DEFAULT_PORT_DATA
        
    # Define a log file for the streamer subprocess.
    filename_log_streamer = "test_proj_pharma_streamer.log"

    with open(filename_log_streamer, "w") as file_log_streamer:
        server_process = subprocess.Popen(["python", "sim_data_streamer.py"],
                                          stdout = file_log_streamer, 
                                          stderr = subprocess.STDOUT,
                                          universal_newlines = True)


    # proj = aml.AutonoMachine()
    # port = proj.ingest_stream(server_hostname, server_port_data, in_id_stream = "abrupto")
    
    # print("USER: Notices the streaming server is activating.")
    # await aml.user_pause(5)

    # print("USER: Inspects data port for its automated field names.")
    # field_names = port.get_field_names()
    # print(field_names)
    # await aml.user_pause(5)
    
    # print("USER: Renames the column names and activates storage.")
    # port.set_field_names(["X1", "X2", "X3", "X4", "class"])
    # port.toggle_storage()
    # await aml.user_pause(10)

    # print("USER: Examines data in storage.")
    # proj.info_storage()
    # await aml.user_pause(5)
    
    # print("USER: Decides on a machine learning task.")
    # proj.learn("class")
    # await aml.user_pause(5)
    
    # print("USER: Stops the AutonoMachine.")
    # proj = None

    # An unconnected server should die eventually, but terminate it anyway.
    server_process.kill()