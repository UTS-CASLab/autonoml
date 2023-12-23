# AutonoML

A framework for continuous automated machine learning.

## Installation

### Prerequisites

- **Anaconda**: If you don't have Anaconda installed, follow these instructions to [install Anaconda](https://www.anaconda.com/products/distribution).
- **Python**: Ensure that you have Python installed. You can check your Python version by running:
  ```bash
  python --version
  ```

### Procedure

1. Clone the repository:
  ```bash
  git clone https://github.com/UTS-CASLab/autonoml.git
  ```

2. Using Anaconda Prompt, navigate to the root directory of the cloned repository, i.e. where setup.py exists.

3. Create and activate a virtual environment. (Note: The current version of the code runs on Python 3.11.)
  ```bash
  conda create --name autonoml python=3.11
  
  conda activate autonoml
  ```

4. Install the AutonoML package. In the root folder (where setup.py is) run:
  ```bash
  pip install -e .
  ```

## For Users

### Basics

The AutonoML paradigm is intended to support continuous learning with flexible user interactivity.
It is thus recommended that AutonoML scripts are run with IPython, e.g. Jupyter, so that users can inspect and manipulate code objects.
However, scripts can also work with standard Python.

Every Python script that uses the AutonoML package should have the form:
  ```python
  import autonoml as aml
  
  if __name__ == '__main__':
  
      proj = aml.AutonoMachine()
  
	  # Do stuff with the AutonoMachine.
  ```

An AutonoMachine is the central object of the framework and wraps around every ML operation.

Note that the moment a script imports the 'autonoml' package, a .log file with the same name as the .py script will be generated.

### Data Ingestion

Typically, the first step of AutonoML is to ingest data.

In AutonoML space, data comes in two forms: observations and queries.
Observations are used to develop ML models, e.g. training/adaptation, while queries are used only to generate responses.
For traditionally static ML, data used to test ML models is considered as queries; this data does not develop ML models.

#### From File

Once an AutonoMachine is instantiated, you can ingest observations and queries from .csv files as follows:
  ```python
  proj.ingest_file(in_filepath = "./data/dummy/train_dummy.csv")
  proj.query_with_file(in_filepath = "./data/dummy/test_dummy.csv")
  ```
You should replace the filepaths as required; the example uses the path of example datasets that are relative to scripts in the tests subfolder.

Once ingested, you can examine some details about the data with:
  ```python
  proj.info_storage()
  ```
Expert users can alternatively explore `proj.data_storage.observations` and `proj.data_storage.queries`, which are dictionaries of DataCollection objects.

#### From Stream

##### Server

You can simulate a stream from a .csv file by launching a SimDataStreamer object as follows:
  ```python
  streamer = aml.SimDataStreamer(in_filepath_data = filepath_data, 
                                 in_period_data_stream = period_data_stream,
                                 in_delay_before_start = delay_before_start,
                                 in_hostname_observations = hostname_observations,
                                 in_port_observations = port_observations)
  ```
You will need to provide the filepath of the .csv file, as well as how often instances of data should be broadcast, i.e. the periodic minimum number of seconds between subsequent instances.
A hostname and port number at which the observations should be broadcasted are also required.
For example, when locally simulating the stream, you can use "127.0.0.1" and "50001".

Given that the streamer acts as a broadcasting server, connecting a client to a server can often have a few seconds of start-up lag.
Thus, if you do not want to miss any instances, a delay before broadcasting can be added.

Often, it can be useful to simulate the broadcaster on its own process.
In the tests subfolder, there is a sim_stream.py file that creates a SimDataStreamer and can be launched with the subprocess module:
  ```python
  import subprocess
  
  with open("./stream.log", "w") as file_log_streamer:
      server_process = subprocess.Popen(["python", "sim_stream.py", 
                                         "--filepath_data", filepath_data, 
                                         "--period_data_stream", str(period_data_stream),
                                         "--delay_before_start", str(delay_before_start),
                                         "--hostname_observations", hostname_observations,
                                         "--port_observations", str(port_observations)],
                                        stdout = file_log_streamer,
                                        stderr = subprocess.STDOUT,
                                        universal_newlines = True)
  ```
Make sure to update the filepaths appropriately as these are relative to the tests subfolder.

##### Client

Ingesting the stream is similar to ingesting a file:
  ```python
  proj.ingest_stream(hostname_observations, port_observations, 
                     in_id_stream = "some_name_for_my_stream", 
                     in_field_names = field_names)
  ```
Ensure the hostname and port number are identical, and choose a name for the stream.

Note that ingesting a standard .csv file automatically stores the headers, i.e. data field names.
In the case of a simulated data stream, no field names are provided by the broadcaster.
So, ensure you provide a list of field names that matches the incoming fields of data prior to activating data ingestion.

Once the AutonoMachine client connects to the broadcasting SimDataStream server, it will regularly send connection confirmations (provided that its process is not clogged with other operations).
If the SimDataStreamer does not receive a confirmation from any client for long enough, it will close itself.

### Learning

Once you have ingested or are in the process of ingesting data, it is time to specify an ML problem and seek an ML solution.

At its most complex, an ML solution is an ensemble of learner groups, where every learner is called an ML pipeline.
An ML pipeline is a sequential structure of ML components that may be ML preprocessors, e.g. a scaler, and ML predictors, e.g. a linear regressor.
In the simplest case, an ML pipeline is simply a single predictor.

#### Strategy

To seek out an ML solution to an ML problem, the AutonoMachine needs to have a strategy for tackling the learning process.
This is done with a strategy file, a template of which can be generated via:
  ```python
  aml.template_strategy(in_filepath = "./template.strat")
  ```


