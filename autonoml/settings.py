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

    #%% Streamer-specific constants.
    DEFAULT_HOSTNAME = "localhost"  # Default IP address for connections.
    DEFAULT_PORT_DATA = 50001       # Ephemeral port within IANA-advised range.
    DEFAULT_PORT_QUERY = 50002      # Ephemeral port within IANA-advised range.
    PERIOD_DATA_STREAM = 1          # Seconds between streamed data instances.
    
    # The following manages when the data-streaming server knows when to shut down.
    PERIOD_SHUTDOWN_CHECK = 20          # Seconds between server checks to see if any client is connected.
    DELAY_FOR_SHUTDOWN_CONFIRM = 30     # Seconds beyond which a server without clients shuts down.
    DELAY_FOR_CLIENT_CONFIRM = 10       # Seconds until client confirms socket is still connected.
    DELAY_FOR_SERVER_ABANDON = 20       # Seconds until server abandons a socket without client confirm.
    SIGNAL_CONFIRM = "1\n"              # An endline-terminated signal used to confirm connection.