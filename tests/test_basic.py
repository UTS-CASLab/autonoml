"""
Created on Tue Apr  4 16:16:30 2023

@author: David J. Kedziora
"""

import autonoml as aml
import asyncio

# import subprocess

# asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def test():
    
    # Make checks more frequent for this test.
    aml.SystemSettings.BASE_DELAY_FOR_ISSUE_CHECK = 1
    
    # Define a default data server host/port for the user to connect with.
    server_host = aml.SystemSettings.DEFAULT_HOST
    server_port = aml.SystemSettings.DEFAULT_PORT_DATA
    
    # async def print_process_output:
        
    # #     for line in p.stdout:
    # #         print(line)
    # subprocess.Popen(["python", "sim_data_streamer.py"],
    #                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #     for line in p.stdout:
    #         print(line, end='')
    # asyncio.get_event_loop().create_task(print_process_output)
    
    # proc = await asyncio.create_subprocess_exec(
    # "python sim_data_streamer.py",
    # stdout=asyncio.subprocess.PIPE,
    # stderr=asyncio.subprocess.PIPE)
    
    # stdout, stderr = await proc.communicate()
    
    # Make a non-blocking task to fake the user pausing during UI interactions.
    user_pause_duration_short = 5
    user_pause_duration_long = 30
    async def user_pause(in_duration):
        await asyncio.sleep(in_duration)
    
    # # Simulate a data-streaming server in its own process.
    # streamer = aml.SimDataStreamer()
    
    print("USER: Launches an AutonoMachine.")
    proj = aml.AutonoMachine()
    
    print("USER: Thinks for %i+ seconds." % user_pause_duration_short)
    task = asyncio.get_event_loop().create_task(user_pause(user_pause_duration_short))
    await task
    
    print("USER: Opens a data port to the streamer.")
    proj.open_data_port(server_host, server_port)
    
    print("USER: Thinks for %i+ seconds." % user_pause_duration_long)
    task = asyncio.get_event_loop().create_task(user_pause(user_pause_duration_long))
    await task
    
    print("USER: Stops the AutonoMachine.")
    proj.stop()
    # proj.open_port_sensor()
    # proj.open_port_actuator()
    # proj.set_objective() predict class
    
    # # Terminate the data-streaming server.
    # streamer.stop()

if __name__ == "__main__":
    # print('output: ', result.stdout)
    # print('error: ', result.stderr)
    
    # Launch the test.
    # Depends on Python environment and whether it already runs an event loop.
    loop = asyncio.get_event_loop()
    if loop.is_running() == False:
        asyncio.run(test())
    else:
        loop.create_task(test())