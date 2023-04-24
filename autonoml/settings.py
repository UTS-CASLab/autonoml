# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:20:57 2023

@author: David J. Kedziora
"""

class SystemSettings:
    """
    A container for all constants used in the AutonoML framework.
    
    Note that most time delays are used in asynchronous tasks.
    Therefore, the real time delays are greater in practice.
    """
    
    BASE_DELAY_FOR_ISSUE_CHECK = 10     # Seconds until checking for issues.

    # Streamer-specific constants.
    DEFAULT_HOST = "localhost"   # Default IP address for connections.
    DEFAULT_PORT = 50001         # Ephemeral port within IANA-advised range.
    PERIOD_DATA_STREAM = 1      # Seconds between streamed data instances.
    DELAY_FOR_SOCKET_CONFIRM = 10       # Seconds until client confirms socket is still connected.
    DELAY_FOR_SOCKET_TIMEOUT = 20       # Seconds until server decides client is not connected.