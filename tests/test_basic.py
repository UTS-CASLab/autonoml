"""
Created on Tue Apr  4 16:16:30 2023

@author: David J. Kedziora
"""

import autonoml as aml
import asyncio

async def test():
    
    # Make checks more frequent for this test.
    aml.SystemSettings.BASE_DELAY_UNTIL_CHECK = 1
    
    # Make a non-blocking task to fake the user pausing during UI interactions.
    user_pause_duration = 5
    async def user_pause(in_duration):
        await asyncio.sleep(in_duration)
    
    print("USER: Launches an AutonoMachine.")
    proj = aml.AutonoMachine()
    
    print("USER: Thinks for %i+ seconds." % user_pause_duration)
    task = asyncio.get_event_loop().create_task(user_pause(user_pause_duration))
    await task
    
    print("USER: Stops the AutonoMachine.")
    proj.stop()
    # proj.open_port_sensor()
    # proj.open_port_actuator()
    # proj.set_objective() predict class

if __name__ == "__main__":
    
    # Launch the test.
    # Depends on Python environment and whether it already runs an event loop.
    loop = asyncio.get_event_loop()
    if loop.is_running() == False:
        asyncio.run(test())
    else:
        loop.create_task(test())