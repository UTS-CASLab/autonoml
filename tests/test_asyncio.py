import autonoml as aml
import asyncio

proj = aml.AutonoMachine()
proj = aml.AutonoMachine()
proj = aml.AutonoMachine()

print("Asynchronous tasks running on main-thread event loop, "
      "if it exists...")
try:
    all_tasks = asyncio.all_tasks()
except:
    all_tasks = None
print(all_tasks)
print("Asynchronous tasks running on alternate-thread event loop, "
      "dedicated to AutonoML...")
print(asyncio.all_tasks(aml.loop_autonoml))