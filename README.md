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
  aml.template_strategy(in_filepath = "./my_strategy.strat")
  ```
Alternatively, run `tests/gen_strategy.py`.\
After constructing the template file, edit it according to its enclosed instructions.

Crucially, either by wholesale inclusions/exclusions or by individual selections, you must select a search space of components suitable for your problem.\
You must also decide whether initial ML pipelines and their defining hyperparameters are selected from this search space in a default manner, randomly, or via hyperparameter optimisation (HPO).\
There are also other choices that can be made, e.g. holdout validation and loss function settings.

Make sure the strategy is appropriate for your intended problem.\
For example, use classifiers (not regressors) and zero-one loss for a classification problem, e.g. `tests/test_proj_iris.strat` for `tests/test_proj_iris.py`.\
Likewise, use regressors (not classifiers) and default RMSE for a regression problem, e.g. `tests/test_proj_sps_static.strat` for `tests/test_proj_sps_static.py`.\
Similarly, consider leaving initial HPO for static ML where sufficient data has already been ingested.

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
So, `do_immediate_responses = False` is used for static ML, where (testing) queries are not processed until all pipelines are fully trained.\
Do not set this argument as false for continuous learning, as continuous pipeline adaptation may block query processing indefinitely.

Also, note that the learn method should be at the end of any script run by non-interactive Python, as, in that setting, it will block the primary thread while AutonoML operations are running (possibly even forever until forcibly broken).

##### Understanding It

When a user requests the AutonoMachine to learn, it creates an object called a ProblemSolver, which can be inspected via `proj.solver`.\
The ProblemSolver itself creates an object called a ProblemSolution, which can be inspected via `proj.solver.solution`.\
Based on the learning instructions, this ML solution is initialised with pre-defined space for learner groups, i.e. champion and challenger pipelines.\
In shorthand, the champion of a group is referred to as L0, while challengers are L1, L2, and so on.

The initial development phase of learning involves selecting ML pipelines from a search space, according to the user's strategy file, and putting them on a development queue.\
These pipelines are trained on whatever observations are immediately available and are then ranked by a validation metric.\
The best become champions and the rest fill out as many challenger slots as are available.

In the simplest case, there is only one learner group. To inspect the ML pipelines of this group, from best to worst, run:
  ```python
  proj.solver.solution.get_learners()
  ```

Every subsequent observation that enters data storage after the initial development phase, e.g. via ongoing streaming, triggers adaptation for every available learner in the ProblemSolution, if applicable.\
Part of adaptation involves validating pipelines on the new observations, and these updated loss values reshuffle the ranks of the champion and challengers.

#### Outputs

All AutonoMachine runs should generate a results subfolder, where, at a bare minimum, `results/info_pipelines.txt` will map the name of every learner that ever became a champion/challenger to its pipeline structure and hyperparameters.\
To actually export all the pipelines presently in a ProblemSolution, run:
  ```python
  proj.solver.solution.export_learners()
  ```
This will pickle the pipelines via the joblib module and save them in a pipelines subfolder, so you can re-import each file via:
  ```python
  import joblib
  
  pipeline = joblib.load(filepath)
  ```
In the static ML case, i.e. where `do_immediate_responses = False`, pipelines will automatically be exported before responding to queries.

Now, if the AutonoMachine run involves any queries, the results subfolder will also contain `results/responses.csv`.\
This output file will contain a unique ID for every query, listing out all the features (with an 'F_' prefix) and, if available, the target (with a 'T_' prefix).\
The output file will also provide a solution prediction (with an 'S_' prefix), which, by default, will be the mean of all champion predictions or, failing that, the mode.\
If there is only one learner group, the ProblemSolution prediction should be equivalent to the solitary champion's prediction.

The predictions of L0, L1, and so on, are included in the output file. So is the name of the pipeline that occupied the learner slot at the time of the response, as well as its loss value (if available/calculated) over the most recent batch of responses.\
In streamed cases, expect the champions and challengers to regularly change.\
Note that a champion can easily have a worse testing loss than a challenger. This just means it was ranked inaccurately, e.g. the training-set validation loss is not representative of the test set.

Now, if the AutonoMachine run involves any streamed observations, the results subfolder will also contain `results/dynamics.csv`.\
This output file is essentially the same as for responses, only it tracks how learners fare with any non-initial observations, i.e. streamed data.\
Unlike with responses, the losses shown here do impact learner rankings.\
Warning: Be aware that 'testing' losses are currently calculated in cumulative fashion across non-initial observations and queries. Be careful when mixing static and continuous ML.

### Advanced Usage

It is possible to attach 'tags' to observations/queries from files and streams. For example, see `tests/test_tags.py`.\
For another example, consider the following:
  ```python
  for a in range(2):
      for b in range(3):
          proj.ingest_file(in_filepath = "data_%s%s.csv" % (a, b),
                           in_tags = {"a": a, "b": b})
  ```

It is then possible to allocate multiple learner groups to different subsets of data, rather than simply having a single learner group. This allows for ML scenarios with specialised experts.\
For instance, consider the following:
  ```python
  proj.learn(in_key_target = "my_target",
             in_strategy = my_strategy,
             in_tags_allocation = ["a", ("b", aml.AllocationMethod.LEAVE_ONE_OUT)])
  ```
This will create 6 learner groups for this example, for which the group keys can be inspected via `proj.solver.solution.groups.keys()`.\
In this example, the champion/challengers of group '(a==1)&(b!=2)' will train/adapt on data_10.csv and data_11.csv.\
To get the learners of this group, run:
  ```python
  proj.solver.solution.get_learners(in_key_group = "(a==1)&(b!=2)")
  ```

However, in general, the process stays the same.\
Note that, with multiple learner groups, there will be one output dynamics file per tag combination.\
For output response files, there will be one per query tag combination, but, at present, every learner group makes a prediction for each query.

## For Developers

### New Components

When the autonoml package is first imported, a catalogue of components in the package is automatically generated by examining every python file in the autonoml/components subfolder. It can be examined with `aml.catalogue.components`.

To add your own custom ML predictor, create/extend a .py file in the subfolder with the following code:
  ```python
from ..hyperparameter import HPCategorical, HPInt, HPFloat
from ..component import (MLPreprocessor, MLPredictor, 
                         MLOnlineLearner, MLDummy,
                         MLImputer, MLScaler, 
                         MLClassifier, MLRegressor)
from ..data import DataFormatX, DataFormatY

from custom_package import custom_model

class MyCustomComponent(MLPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = custom_model(hp_1 = self.hpars["HP_1"].val,
                                  hp_2 = self.hpars["HP_2"].val,
                                  hp_3 = self.hpars["HP_3"].val)
        self.name += "_MyCoolModel"
        self.format_x = DataFormatX.NUMPY_ARRAY_2D
        self.format_y = DataFormatY.NUMPY_ARRAY_2D

    @staticmethod
    def new_hpars():
        hpars = dict()
        info = ("HP_1 is a fake categorical hyperparameter made up for this example.")
        hpars["HP_1"] = HPCategorical(in_default = "opt_2", in_options = ["opt_1", "opt_2", "opt_3"],
                                      in_info = info)
        hpars["HP_2"] = HPInt(in_default = 5, in_min = 0, in_max = 10)
        hpars["HP_3"] = HPFloat(in_default = 0.01, in_min = 0.0001, in_max = 1.0,
                                is_log_scale = True)
        return hpars

    def learn(self, x, y):
        self.model.custom_fit(x=x, y=y)

    def query(self, x):
        return self.model.custom_predict(x=x)
  ```
Essentially, your new class must inherit MLPredictor, which in turns inherits MLComponent.\
It must wrap around your custom model with the model attribute, optionally initialising it with hyperparameters that are defined in the `new_hpars()` method.\
Predictors must also implement a learn and query method.

Importantly, a 2D table of features, i.e. x, and a 1D column of a target variable, i.e. y, can have many different formats.\
The autonoml package handles conversions in the background, but you still need to specify the format of x and y.\
Examine `autonoml/data.py` for what is available, running `tests/test_data_conversions.py` for further information.

Now, if you want to add an ML preprocessor, you need to implement a transform method rather than a query method.
  ```python
class MyCustomComponent(MLPreprocessor):
    ...
    
    def transform(self, x):
        return self.model.custom_transform(x=x)
  ```

By default, both predictors and preprocessors are open to adaptation but do not functionally adapt.\
You can implement a custom adapt method. Note that any incremental learner must first break up the batched x and y into increments, if necessary.\
For example, in abstract terms:
  ```python
    def adapt(self, x, y):
        for x_i, y_i in zip(x, y):
            self.model.custom_incremental_fit(x=x_i, y=y_i)
  ```
Note that a predictor or preprocessor can multiply inherit an MLOnlineLearner to, by default, make the adapt method redirect to the learn method, as the two are usually identical for an incremental learner.\
Any other multiple inheritances, e.g. of MLDummy, are simply to aid with inclusions/exclusions in strategy files.

## Ideal Roadmap