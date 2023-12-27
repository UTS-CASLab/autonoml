"""
Created on Tue Apr  4 16:16:30 2023

@author: David J. Kedziora
"""

import autonoml as aml

import asyncio
import subprocess

async def test():
    
    # Make checks more frequent for this test.
    aml.SystemSettings.BASE_DELAY_FOR_ISSUE_CHECK = 1.0
    
    # Define a default data server host/port for the user to connect with.
    server_hostname = aml.SystemSettings.DEFAULT_HOSTNAME
    server_port_observations = aml.SystemSettings.DEFAULT_PORT_OBSERVATIONS
        
    # Name the data/log file for the streamer subprocess as well as the broadcasting period.
    filepath_data_streamer = "./data/drift/mixed_0101_abrupto.csv"
    filename_log_streamer = "./test_streaming_setup_streamer.log"
    period_data_stream = 1.0
    
    
    
    print("USER: Launches an AutonoMachine.")
    proj = aml.AutonoMachine()
    await aml.user_pause(5)
    
    print("USER: Opens a data port to an inactive streaming server.\n"
          "      They are unconcerned by connection errors.")
    port = proj.ingest_stream(server_hostname, server_port_observations, in_id_stream = "abrupto")
    await aml.user_pause(10)
    
    print("USER: Notices the streaming server is activating.\n"
          "      They are unconcerned by initial connection errors.")
    with open(filename_log_streamer, "w") as file_log_streamer:
        server_process = subprocess.Popen(["python", "sim_stream.py", 
                                           "--filepath_data", filepath_data_streamer, 
                                           "--period_data_stream", str(period_data_stream)],
                                          stdout = file_log_streamer, 
                                          stderr = subprocess.STDOUT,
                                          universal_newlines = True)
    await aml.user_pause(5)

    print("USER: Inspects data port for its automated field names.")
    field_names = port.get_field_names()
    print(field_names)
    await aml.user_pause(5)
    
    print("USER: Renames the column names and activates storage.")
    port.set_field_names(["X1", "X2", "X3", "X4", "class"])
    port.toggle_storage()
    await aml.user_pause(10)

    print("USER: Examines data in storage.")
    proj.info_storage()
    await aml.user_pause(5)
    
    print("USER: Decides on a machine learning task.")
    proj.learn("class")
    await aml.user_pause(5)
    
    print("USER: Deletes the AutonoMachine.")
    proj = None
    
    # An unconnected server should die eventually, but terminate it anyway.
    server_process.kill()

if __name__ == "__main__":
    
    # Launch the test.
    # Depends on Python environment and whether it already runs an event loop.
    task = None
    loop = asyncio.get_event_loop()
    if loop.is_running() == False:
        task = asyncio.run(test())
    else:
        task = loop.create_task(test())