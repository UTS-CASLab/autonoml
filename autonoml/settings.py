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
    
    BASE_DELAY_FOR_ISSUE_CHECK = 10.0   # Seconds until checking for issues.

    #%% Logger settings.
    LOG_HPO_OPTIMISER = False
    LOG_HPO_WORKER = False

    #%% Streamer-specific constants.
    DEFAULT_HOSTNAME = "localhost"          # Default IP address for connections.
    DEFAULT_PORT_OBSERVATIONS = 50001       # Ephemeral port within IANA-advised range.
    DEFAULT_PORT_QUERIES = 50002            # Ephemeral port within IANA-advised range.
    PERIOD_DATA_STREAM = 1.0                # Seconds between streamed data instances.
    
    # The following manages when the data-streaming server knows when to shut down.
    PERIOD_SHUTDOWN_CHECK = 20.0        # Seconds between server checks to see if any client is connected.
    DELAY_BEFORE_START = 0.0            # Seconds before simulator starts to generate data to broadcast.
    DELAY_FOR_SHUTDOWN_CONFIRM = 30.0   # Seconds beyond which a server without clients shuts down.
    DELAY_FOR_CLIENT_CONFIRM = 10.0     # Seconds until client confirms socket is still connected.
    DELAY_FOR_SERVER_ABANDON = 20.0     # Seconds until server abandons a socket without client confirm.
    SIGNAL_CONFIRM = "1\n"              # An endline-terminated signal used to confirm connection.
    MAX_ATTEMPTS_RECONNECTION = 1       # How many times a once-connected client will retry a connection.
    
    # #%% IO settings.
    # MAX_ALERTS_IKEY_NEW = 5     # How many newly encountered DataPort inflow keys to individually acknowledge.
    # MAX_ALERTS_DKEY_NEW = 5     # How many newly established DataStorage data keys to individually acknowledge.
    # MAX_INFO_KEYS_EXAMPLE = 10  # How many DataStorage keys to exemplify when providing info.
    # MAX_INFO_PIPE_EXAMPLE = 10  # How many inflow-to-data pipes to exemplify when providing info.

    #%% Solver settings.
    MAX_ATTEMPTS_DEVELOPMENT = 5    # How many times a failed development package will return to the queue.
    BASE_DELAY_BEFORE_RETRY = 2.0   # Baseline seconds to wait before retrying a failed development.
    
    #%% Plot settings.
    MAX_LABELS_BAR = 25     # If there are more bars in a chart, label a fraction of them.
    BINS_HIST = 100         # Number of bins for a histogram to show.

    #%% Hyperparameter settings.
    INT_MIN = -int(1e9)
    INT_MAX = int(1e9)
    FLOAT_MIN = -float(1e9)
    FLOAT_MAX = float(1e9)