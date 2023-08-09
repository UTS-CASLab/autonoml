# -*- coding: utf-8 -*-
"""
This script tests that AutonoMachine-started asyncio tasks operate properly.
Specifically, they must be scheduled on the appropriate loop.
This script must run in both Python and IPython.
Additionally, tasks must be cancelled if their callers fall out of scope.
TODO: Wait for the task scheduler to fully complete before inspecting the loop.

Created on Thu Aug  3 18:59:35 2023

@author: David J. Kedziora
"""

import autonoml as aml
import asyncio

proj_1 = aml.AutonoMachine()
proj_2 = aml.AutonoMachine()
proj_3 = aml.AutonoMachine()
aml.inspect_loop()
proj_1, proj_2, proj_3 = None, None, None

proj = aml.AutonoMachine()
proj = aml.AutonoMachine()
proj = aml.AutonoMachine()
aml.inspect_loop()
proj = None