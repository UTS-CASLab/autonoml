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

2. Using Anaconda Prompt, navigate to the root directory of the cloned repository, i.e. where setup.py exists.\
   Take note of the autonoml subfolder, where the actual code sits, and the tests subfolder, where useful example scripts exist.

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

The AutonoML paradigm is intended to support static/continuous machine learning (ML) with flexible user interactivity.\
It is recommended that AutonoML scripts are run with IPython, e.g. Jupyter, so that users can inspect and manipulate code objects during continuous processing. 
However, scripts should also work with standard Python.

Every Python script that uses the AutonoML package should have the form:
  ```python
  import autonoml as aml
  
  if __name__ == '__main__':
  
      proj = aml.AutonoMachine()
  
	  # Do stuff with the AutonoMachine.
  ```

An AutonoMachine is the central object of the framework and wraps around every ML operation.

Note that the moment a script imports the autonoml package, a .log file with the same name as the .py script will be generated.\
Inspect these for further details of any autonoml run.\
Warning: During standard multi-threaded runs, logged print-outs can interleave.

### Data Ingestion

Typically, the first step of AutonoML is to ingest data. The current version only supports tabular data.

In AutonoML space, data comes in two forms: observations and queries.\
Observations, i.e. instances of 'ground truth', are used to develop ML models, e.g. training/adaptation.\
Queries are used to generate responses without model development. However, they can contain expected responses. In fact, test data for static ML should be presented to the AutonoMachine as queries.

#### From File

Once an AutonoMachine is instantiated, you can ingest observations and queries from .csv files as follows:
  ```python
  proj.ingest_file(in_filepath = "./data/dummy/train_dummy.csv")
  proj.query_with_file(in_filepath = "./data/dummy/test_dummy.csv")
  ```
Replace the filepaths as required; the example uses the path of example datasets that are relative to scripts in the tests subfolder.\
Also, add an `in_n_instances = 100` argument to only store the first 100 instances from file.

Once ingested, you can examine some details about the data with:
  ```python
  proj.info_storage()
  ```
Expert users can alternatively explore `proj.data_storage.observations` and `proj.data_storage.queries`, which are dictionaries of DataCollection objects.

#### From Stream

For examples on how to work with streamed data, examine and run `tests/test_streaming_setup.py` and `tests/test_proj_pharma.py`.

##### Server

You can simulate a stream from a .csv file by launching a SimDataStreamer object as follows:
  ```python
  streamer = aml.SimDataStreamer(in_filepath_data = filepath_data, 
                                 in_period_data_stream = period_data_stream,
                                 in_delay_before_start = delay_before_start,
                                 in_hostname_observations = hostname_observations,
                                 in_port_observations = port_observations)
  ```
Provide the filepath of the .csv file, as well as how often instances of data should be broadcast, i.e. the data stream period in seconds. This will be the minimum wait time between sequential broadcasts.\
A hostname and port number for the broadcast server are also required. For example, when locally simulating the stream, you can use "127.0.0.1" and "50001".\
Note that connecting a client to this server can often have a few seconds of start-up lag. If you do not want the client to miss any instances, a delay before broadcasting can be added.

Now, it can often be useful to simulate the broadcaster on its own process.\
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
Update the filepaths appropriately as these are relative to the tests subfolder.

##### Client

Ingesting the stream is similar to ingesting a file:
  ```python
  proj.ingest_stream(hostname_observations, port_observations, 
                     in_id_stream = "some_name_for_my_stream", 
                     in_field_names = field_names)
  ```
Ensure the hostname and port number are identical between AutonoMachine and SimDataStreamer, and choose a name for the stream.

Note that ingesting a standard .csv file automatically stores the headers, i.e. data field names.\
In the case of a simulated data stream, no field names are provided by the broadcaster.\
So, ensure you provide a list of field names that matches the incoming fields of data prior to activating data ingestion.\
See `tests/test_streaming_setup.py` for updating and activating after connection. See `tests/test_proj_pharma.py` for specifying field names as part of the connection, and thus auto-activating data storage.

Once the AutonoMachine client connects to the broadcasting SimDataStream server, it will regularly send connection confirmations (provided that its process is not clogged with other operations).\
If the SimDataStreamer does not receive a confirmation from any client for long enough, it will close itself.

### Learning

Once data is ingested or in the process of being ingested, it is time to specify an ML problem and seek an ML solution.

At its most complex, an ML solution is an ensemble of learner groups, where every learner is called an ML pipeline.\
An ML pipeline is a sequential structure of ML components that may be ML preprocessors, e.g. a scaler, and ML predictors, e.g. a linear regressor.\
In the simplest case, an ML pipeline is a single predictor.

#### Strategy

To seek out an ML solution to an ML problem, the AutonoMachine needs to have a strategy for tackling the learning process.\
This is done with a strategy file, a template of which can be generated via:
  ```python
  aml.template_strategy(in_filepath = "./template.strat")
  ```
Alternatively, run `tests/gen_strategy.py`.\
After constructing the template file, edit it according to its enclosed instructions.

Crucially, either by wholesale inclusions/exclusions or by individual selections, you must select a search space of components suitable for your problem.\
You must also decide whether initial ML pipelines and their defining hyperparameters are selected from this search space in a default manner, randomly, or via hyperparameter optimisation (HPO).\
There are also other choices that can be made, e.g. for holdout validation and loss function.

Make sure the strategy is appropriate for your intended problem.\
For example, use classifiers (not regressors) and zero-one loss for a classification problem, e.g. `tests/test_proj_iris.strat` for `tests/test_proj_iris.py`.\
Likewise, use regressors (not classifiers) and default RMSE for a regression problem, e.g. `tests/test_proj_sps_static.strat` for `tests/test_proj_sps_static.py`.\
Additionally, HPO is likely to be used only for static ML.

You can see what components are in the search space for your edited/renamed strategy file via:
  ```python
  strategy = aml.import_strategy("./my_strategy.strat")
  ```

#### The Process

##### Launching It

Once the data is stored, the learning process can be launched with:
  ```python
  proj.learn(in_key_target = "my_target",
             in_keys_features = ["feature_a", "feature_b"], do_exclude = True,
             do_immediate_responses = False,
             in_strategy = aml.import_strategy("./my_strategy.strat"))
  ```
It is minimally necessary to provide a target field to learn, as well as an imported strategy file (as the default does not choose a search space or pipeline selection method).\
You can also specify what features to include or exclude. If the option is to exclude, every other feature that presently exists in data storage will be part of the learning process.

AutonoML is, by default, a continuous learner. It will respond to queries immediately as they arrive, regardless of what the ML solution looks like at the time.\
So, `do_immediate_responses = False` is used for static ML, where (testing) queries are not processed until every selected pipeline is fully trained.

Also, note that the learn method should be at the end of any script run by non-interactive Python, as, in that setting, it will block the primary thread while AutonoML operations are running (possibly even forever until forcibly broken).

##### Understanding It

When a user requests the AutonoMachine to learn, it creates an object called a ProblemSolver, which can be inspected via `proj.solver`.\
The ProblemSolver itself creates an object called a ProblemSolution, which can be inspected via `proj.solver.solution`.\
Based on the learning instructions, the ML solution is initialised with space for learner groups, i.e. champion and challenger pipelines.

The initial development phase of learning involves selecting ML pipelines from a search space, according to the user's strategy file, and putting them on a development queue.\
These pipelines are trained on whatever observations are immediately available and are then ranked by a validation metric.\
The best become champions and the rest fill out as many challenger slots as are available.

In the simplest case, there is only one learner group. To inspect the ML pipelines of this group, from best to worst, run:
  ```python
  proj.solver.solution.get_learners()
  ```


