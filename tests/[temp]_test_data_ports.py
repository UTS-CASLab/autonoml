# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:08:54 2023

@author: David J. Kedziora
"""

import autonoml as aml

import asyncio
import subprocess

async def test():
    
    # Make checks more frequent for this test.
    aml.SystemSettings.BASE_DELAY_FOR_ISSUE_CHECK = 1
    
    # Define two data servers by host/port for the user to connect with.
    server_hostname = aml.SystemSettings.DEFAULT_HOSTNAME
    server_port_data_1 = aml.SystemSettings.DEFAULT_PORT_DATA
    server_port_query_1 = aml.SystemSettings.DEFAULT_PORT_QUERY
    server_port_data_2 = aml.SystemSettings.DEFAULT_PORT_DATA + 10
    server_port_query_2 = aml.SystemSettings.DEFAULT_PORT_QUERY + 10
        
    # Define a log file for the streamer subprocesses.
    filename_log_streamer_1 = "test_data_ports_streamer_1.log"
    filename_log_streamer_2 = "test_data_ports_streamer_2.log"
    


    print("USER: Notices two data streaming servers are active.")
    with open(filename_log_streamer_1, "w") as file_log_streamer_1:
        server_process_1 = subprocess.Popen(["python", "sim_data_streamer.py"],
                                            stdout = file_log_streamer_1,
                                            stderr = subprocess.STDOUT,
                                            universal_newlines = True)
    with open(filename_log_streamer_2, "w") as file_log_streamer_2:
        server_process_2 = subprocess.Popen(["python", "sim_data_streamer_alt.py"],
                                            stdout = file_log_streamer_2,
                                            stderr = subprocess.STDOUT,
                                            universal_newlines = True)
    
    print("USER: Launches an AutonoMachine.")
    proj = aml.AutonoMachine()
    await aml.user_pause(5)
    
    print("USER: Opens two data ports to the streaming servers.")
    proj.open_data_port(server_hostname, server_port_data_1, in_id = "abrupto")
    proj.open_data_port(server_hostname, server_port_data_2, in_id = "gradual")
    await aml.user_pause(15)
    
    print("USER: User checks how data is being stored.")
    proj.info_storage()
    await aml.user_pause(5)
    # print(proj.data_storage.get_dataframe())    # Inspect data for debugging.
    
    print("USER: User renames stored data columns from the first streamer.")
    proj.update_storage(["abrupto_0", "abrupto_1", "abrupto_2", "abrupto_3", "abrupto_4"],
                        ["X1", "X2", "X3", "X4", "class"])
    await aml.user_pause(5)
    
    print("USER: User checks how data is being stored.")
    proj.info_storage()
    await aml.user_pause(5)
    # print(proj.data_storage.get_dataframe())    # Inspect data for debugging.
    
    print("USER: User redirects and merges in data from the second streamer.")
    proj.update_storage(["gradual_0", "gradual_1", "gradual_2", "gradual_fake", "gradual_3", "gradual_4"],
                        ["X1", "X2", "X3", "XFAKE", "X4", "class"])
    await aml.user_pause(5)
    
    print("USER: User checks how data is being stored.")
    proj.info_storage()
    await aml.user_pause(5)
    # print(proj.data_storage.get_dataframe())    # Inspect data for debugging.
    
    print("USER: Stops the AutonoMachine.")
    proj.stop()
    
    # Unconnected servers should die eventually, but terminate them anyway.
    server_process_1.kill()
    server_process_2.kill()
    
    return proj

if __name__ == "__main__":
    
    # Launch the test.
    # Depends on Python environment and whether it already runs an event loop.
    task = None
    loop = asyncio.get_event_loop()
    if loop.is_running() == False:
        task = asyncio.run(test())
    else:
        task = loop.create_task(test())
    # Use: proj = task.result()