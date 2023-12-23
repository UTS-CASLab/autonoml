# -*- coding: utf-8 -*-
"""
This script creates a template strategy file.
Designed for users without access to interactive Python.
Users with IPython can however examine the re-imported strategy.

Created on Tue Sep 19 14:25:29 2023

@author: David J. Kedziora
"""

import autonoml as aml

if __name__ == '__main__':

    aml.template_strategy()
    strategy = aml.import_strategy("./template.strat")