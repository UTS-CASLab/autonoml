"""
Created on Tue Apr  4 16:16:30 2023

@author: David J. Kedziora
"""

import autonoml as aml

import asyncio
import subprocess

async def test():
    
    # Make checks more frequent for this test.
    aml.SystemSettings.BASE_DELAY_FOR_ISSUE_CHECK = 1
    
    # Define a default data server host/port for the user to connect with.
    server_hostname = aml.SystemSettings.DEFAULT_HOSTNAME
    server_port_data = aml.SystemSettings.DEFAULT_PORT_DATA
        
    # Define a log file for the streamer subprocess.
    filename_log_streamer = "test_basic_streamer.log"
    
    
    
    print("USER: Launches an AutonoMachine.")
    proj = aml.AutonoMachine()
    await aml.user_pause(5)
    
    print("USER: Opens a data port to an inactive streaming server.")
    proj.open_data_port(server_hostname, server_port_data, in_id = "abrupto")
    await aml.user_pause(15)
    
    print("USER: Notices the streaming server is activating.")
    with open(filename_log_streamer, "w") as file_log_streamer:
        server_process = subprocess.Popen(["python", "sim_data_streamer.py"],
                                          stdout = file_log_streamer, 
                                          stderr = subprocess.STDOUT,
                                          universal_newlines = True)
    await aml.user_pause(15)
    
    print("USER: User renames stored data columns from the streamer.")
    proj.update_storage(["abrupto_0", "abrupto_1", "abrupto_2", "abrupto_3", "abrupto_4"],
                        ["X1", "X2", "X3", "X4", "class"])
    await aml.user_pause(5)
    
    print("USER: Decides on a machine learning task.")
    proj.learn("class")
    await aml.user_pause(15)
    
    print("USER: Stops the AutonoMachine.")
    proj.stop()
    
    # An unconnected server should die eventually, but terminate it anyway.
    server_process.kill()
    
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