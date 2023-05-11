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
    server_host = aml.SystemSettings.DEFAULT_HOST
    server_port_data = aml.SystemSettings.DEFAULT_PORT_DATA
    
    # Make a non-blocking task to fake the user pausing during UI interactions.
    user_pause_duration_short = 5
    user_pause_duration_medium = 15
    user_pause_duration_long = 45
    async def user_pause(in_duration):
        await asyncio.sleep(in_duration)
        
    # Define a log file for the streamer subprocess.
    filename_log_streamer = "test_basic_streamer.log"
    
    
    
    print("USER: Launches an AutonoMachine.")
    proj = aml.AutonoMachine()
    
    print("USER: Thinks for %i+ seconds." % user_pause_duration_short)
    task = asyncio.get_event_loop().create_task(user_pause(user_pause_duration_short))
    await task
    
    print("USER: Opens a data port to an inactive streaming server.")
    proj.open_data_port(server_host, server_port_data)
    
    print("USER: Thinks for %i+ seconds." % user_pause_duration_medium)
    task = asyncio.get_event_loop().create_task(user_pause(user_pause_duration_medium))
    await task
    
    print("USER: Notices the streaming server is activating.")
    with open(filename_log_streamer, "w") as file_log_streamer:
        server_process = subprocess.Popen(["python", "sim_data_streamer.py"],
                                          stdout = file_log_streamer, 
                                          stderr = subprocess.STDOUT,
                                          universal_newlines = True)
    
    print("USER: Thinks for %i+ seconds." % user_pause_duration_medium)
    task = asyncio.get_event_loop().create_task(user_pause(user_pause_duration_medium))
    await task
    
    # print("USER: Decides on a machine learning task.")
    # proj.learn("class")
    
    print("USER: Thinks for %i+ seconds." % user_pause_duration_medium)
    task = asyncio.get_event_loop().create_task(user_pause(user_pause_duration_medium))
    await task
    
    # print("USER: Thinks for %i+ seconds." % user_pause_duration_long)
    # task = asyncio.get_event_loop().create_task(user_pause(user_pause_duration_long))
    # await task
    
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